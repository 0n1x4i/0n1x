"""Onyx H3 Seamless Sampler — long MiniMax H3 video with invisible seams.

Own Onyx implementation (reuses the vendored Endless Sampler helpers, Apache 2.0).

Three ideas, all prompt-agnostic:

1. Native extension, not a reference. The last ``context_frames`` of the
   previous segment (video AND audio latents) are pinned as the opening
   keyframe of the next one, exactly like a native H3 extension. H3 never sees
   the previous segment as a <Video>/<Audio> source to copy, so it does not
   replay motion or repeat speech.

2. Silence-aware handover. After each segment the audio is decoded and the
   handover point is moved back to the latest moment where nobody is speaking
   (on H3's 17-frame phase grid). A seam can never cut a word in half, whatever
   the prompt says or however H3 timed the line. Needs ``audio_vae``.

3. Runtime timeline. Every timed beat of the prompt (``At MM:SS.mmm,`` and
   ``[Shot N]``) is scheduled into exactly one segment, in order, and is only
   marked done once it really happened before the chosen handover. A beat cut
   away by an early handover is simply scheduled again in the next segment —
   never written twice, never lost. No LLM, nothing rewritten.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import torch

import comfy.model_management
import comfy.nested_tensor
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from ..onyx_render_profile import ensure_profile_ready as _ensure_profile_ready
from .nodes import (
    DESCRIPTION_END,
    SUMMARY_FIELD,
    _audio_steps,
    _conditioning_for_chunk,
    _description_field,
    _encode_prompt,
    _frame_timestamp,
    _parse_prompt_shots,
    _pixel_frames,
    _timestamp_frame,
    _video_steps,
)

LOG = "[Onyx Seamless]"

BEAT_MARKER = re.compile(r"\bAt\s+(\d+):(\d{2})\.(\d{3}),", re.IGNORECASE)
DIALOGUE = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)

BEAT_SPACING_FRAMES = 12          # min gap between two scheduled beats
SPEECH_GAP_FRAMES = 12            # silence kept after a line before the next one
SPEECH_MARGIN_FRAMES = 8          # a line must end this long before the next guide
DEFER_OFFSET_FRAMES = 4           # first beat after the pinned guide frames
WINDOW_PAD_FRAMES = 3             # extra frames checked before the guide window

CONTINUE_PREFIX = (
    "This segment continues the same single continuous take directly from its opening frames: same camera "
    "position, same framing, same lighting, same subject placement. Nothing that already happened is "
    "replayed and nothing already said is repeated. The only speech in this segment is the exact <d> lines "
    "written below; outside them everyone is silent."
)
CONTINUE_IDLE = (
    "The ongoing movement simply continues naturally from the opening frames until the end of this "
    "segment, with no new action and no speech."
)


# ─────────────────────────────────────────────────────────────────────────────
# Timeline (pure python, no ComfyUI state)
# ─────────────────────────────────────────────────────────────────────────────

def speech_frames(text, fps):
    """Rough spoken length of the <d> lines in a beat (2.5 words/s + 0.6 s)."""
    words = 0
    for line in DIALOGUE.findall(text):
        inner = re.sub(r"</?d>|\[[^\]]*\]", " ", line, flags=re.IGNORECASE)
        words += len(inner.split())
    return int(round((0.6 + words / 2.5) * fps)) if words else 0


@dataclass
class Beat:
    frame: int
    text: str
    shot: int
    opens_shot: bool
    speech: int = 0


@dataclass
class Timeline:
    prompt: str
    fps: float
    total_frames: int
    header: str = ""
    desc_start: int = 0
    desc_end: int = 0
    has_markers: bool = True
    pending: list = field(default_factory=list)

    max_line_frames: int = 0
    rescheduled: int = 0

    @classmethod
    def parse(cls, prompt, total_frames, fps, max_line_frames=0):
        tl = cls(prompt=prompt, fps=fps, total_frames=total_frames, max_line_frames=max_line_frames)
        found = _description_field(prompt)
        tl.desc_start = found.end() if found is not None else 0
        end_match = DESCRIPTION_END.search(prompt, tl.desc_start)
        tl.desc_end = end_match.start() if end_match is not None else len(prompt)
        markers, shots, _ = _parse_prompt_shots(prompt, total_frames, fps)
        if not markers:
            tl.has_markers = False
            body = prompt[tl.desc_start:tl.desc_end]
            tl._add_body(body, 0, 0)
            return tl
        tl.header = prompt[tl.desc_start:markers[0].start()].strip()
        for shot_index, shot_start, _shot_end, body in shots:
            tl._add_body(body, shot_start, shot_index)
        return tl

    def _add_body(self, body, shot_start, shot_index):
        matches = list(BEAT_MARKER.finditer(body))
        preamble = body[:matches[0].start()] if matches else body
        for n, piece in enumerate(self._split_long_line(preamble.strip())):
            self.pending.append(Beat(shot_start + n, piece, shot_index, n == 0, speech_frames(piece, self.fps)))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            text = body[match.end():end].strip()
            frame = max(shot_start, _timestamp_frame(match.group(1), match.group(2), match.group(3), self.fps))
            for n, piece in enumerate(self._split_long_line(text)):
                self.pending.append(Beat(frame + n, piece, shot_index, False, speech_frames(piece, self.fps)))

    def _split_long_line(self, text):
        """A line longer than one segment can hold is split at its punctuation
        into consecutive beats, so every part can be spoken inside one segment."""
        if not self.max_line_frames or speech_frames(text, self.fps) <= self.max_line_frames:
            return [text]
        match = DIALOGUE.search(text)
        if match is None or len(DIALOGUE.findall(text)) != 1:
            return [text]
        inner = re.sub(r"</?d>", "", match.group(0), flags=re.IGNORECASE).strip()
        tag_match = re.match(r"(\[[^\]]*\])\s*", inner)
        tag = tag_match.group(1) + " " if tag_match else ""
        words_text = inner[tag_match.end():] if tag_match else inner
        sentences = [x.strip() for x in re.split(r"(?<=[.!?;,])\s+", words_text) if x.strip()]
        pieces, current = [], ""
        for sentence in sentences:
            candidate = (current + " " + sentence).strip()
            if current and speech_frames(f"<d>{candidate}</d>", self.fps) > self.max_line_frames:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)
        if len(pieces) < 2:
            return [text]
        before, after = text[:match.start()], text[match.end():]
        speaker = re.search(r"(<Subject\s+\d+>\s*\(S\d+\))", before)
        who = speaker.group(1) if speaker else "She"
        out = [f"{before}<d>{tag}{pieces[0]}</d>"]
        out += [f"{who} continues: <d>{tag}{piece}</d>" for piece in pieces[1:-1]]
        out.append(f"{who} continues: <d>{tag}{pieces[-1]}</d>{after}")
        return out

    def schedule(self, seg_start, seg_frames, guide_frames, next_guide, is_last):
        """Pick the pending beats for one segment. Returns [(local_frame, Beat)]."""
        cursor = 0 if guide_frames == 0 else guide_frames + DEFER_OFFSET_FRAMES
        speech_cursor = cursor   # actions may overlap a line; lines never overlap each other
        action_limit = seg_frames - next_guide
        speech_limit = seg_frames - next_guide - SPEECH_MARGIN_FRAMES
        chosen = []
        for beat in self.pending:
            local = max(beat.frame - seg_start, cursor, speech_cursor if beat.speech else 0)
            if beat.opens_shot and not chosen and guide_frames == 0:
                local = 0
            if not is_last and chosen:
                if beat.speech and local + beat.speech > speech_limit:
                    break
                if local >= action_limit:
                    break
            elif not is_last and not chosen and local >= action_limit:
                break
            if is_last and (local >= seg_frames or (beat.speech and local + beat.speech > seg_frames)):
                break  # the video ends before this beat could play (or finish speaking)
            chosen.append((local, beat))
            cursor = local + BEAT_SPACING_FRAMES
            if beat.speech:
                speech_cursor = local + beat.speech + SPEECH_GAP_FRAMES
        return chosen

    def commit(self, chosen, cut_frame, rms=None, threshold=None, next_guide=0):
        """Beats that really happened before the handover are done; the rest stay pending.

        A spoken beat is only done if speech is actually heard between its slot
        and the handover window: H3 sometimes starts a line late, and a line
        that slipped past the handover must be scheduled again, not lost.
        """
        done = set()
        for local, beat in chosen:
            if local >= cut_frame:
                continue
            if beat.speech and rms is not None and threshold is not None:
                hi = max(local + 1, cut_frame - next_guide)
                heard = rms[local:hi]
                if heard.shape[0] == 0 or float(heard.max()) < threshold:
                    continue
            done.add(id(beat))
        missed = sum(1 for local, beat in chosen if beat.speech and id(beat) not in done)
        self.rescheduled = self.rescheduled + 1 if missed else 0
        self.pending = [beat for beat in self.pending if id(beat) not in done]
        return len(done)

    def segment_prompt(self, chosen, first, multi):
        parts = []
        if self.header:
            parts.append(self.header)
        if not first:
            parts.append(CONTINUE_PREFIX)
        shot_no = 0 if first else 1
        for local, beat in chosen:
            text = beat.text
            if self.has_markers and beat.opens_shot:
                shot_no += 1
                if shot_no == 1:
                    parts.append(f"[Shot 1] {text}".strip())
                else:
                    parts.append(f"[Shot {shot_no}] At {_frame_timestamp(local, self.fps)}, {text}".strip())
            elif not text:
                continue
            elif local > 0:
                parts.append(f"At {_frame_timestamp(local, self.fps)}, {text}")
            else:
                parts.append(text)
        if not first and not chosen:
            parts.append(CONTINUE_IDLE)
        description = "\n".join(p for p in parts if p)
        prompt = (self.prompt[:self.desc_start].rstrip() + "\n" + description + "\n\n"
                  + self.prompt[self.desc_end:].lstrip("\n"))
        if multi:
            prompt = _segment_summary(prompt, first)
        return prompt


def _segment_summary(prompt, first):
    summary = SUMMARY_FIELD.search(prompt)
    if summary is None:
        return prompt
    task = re.match(r"(\[[^]]+\])", summary.group(2).strip())
    tag = task.group(1) + " " if task else ""
    if first:
        text = (f"{tag}Opening segment of one single continuous take. Only the events written in "
                f"detailed_description happen in this segment; the rest of the story happens later.")
    else:
        text = (f"{tag}Continuation segment of one single continuous take. Only the events written in "
                f"detailed_description happen in this segment; everything else already happened earlier.")
    return prompt[:summary.start()] + summary.group(1) + text + prompt[summary.end():]


# ─────────────────────────────────────────────────────────────────────────────
# Silence-aware handover (pure tensor math)
# ─────────────────────────────────────────────────────────────────────────────

def frame_rms(waveform, frames):
    """Per-video-frame RMS of a decoded waveform of any layout."""
    w = waveform.detach().float().cpu()
    sample_axis = max(range(w.ndim), key=lambda axis: w.shape[axis])
    w = w.movedim(sample_axis, -1).reshape(-1, w.shape[sample_axis]).mean(dim=0)
    per = max(1, w.shape[-1] // max(1, frames))
    w = w[: per * frames]
    if w.shape[-1] < per * frames:
        w = torch.nn.functional.pad(w, (0, per * frames - w.shape[-1]))
    return w.view(frames, per).pow(2).mean(dim=1).sqrt()


def choose_handover(rms, seg_frames, guide_frames, next_guide, search_frames, speech_ratio=0.2):
    """Latest phase-valid handover frame whose next-guide window holds no speech.

    Returns (cut_frame, silent, speech_threshold).
    """
    seg_tokens = _video_steps(seg_frames)
    candidates = []
    token = seg_tokens
    while token >= 2:
        if (token - 2) % 5 == 0:
            frame = _pixel_frames(token)
            if frame - next_guide >= 0 and frame - guide_frames >= 17 and frame >= seg_frames - search_frames:
                candidates.append(frame)
        token -= 1
    if rms is None:
        return (candidates[0] if candidates else seg_frames), True, None
    peak = float(torch.quantile(rms, 0.95)) if rms.numel() > 1 else float(rms.max())
    threshold = max(peak * speech_ratio, 1e-4)
    if not candidates:
        return seg_frames, True, threshold
    best, best_level = candidates[0], None
    for frame in candidates:  # latest first
        lo = max(0, frame - next_guide - WINDOW_PAD_FRAMES)
        level = float(rms[lo:frame].max()) if frame > lo else 0.0
        if level < threshold:
            return frame, True, threshold
        if best_level is None or level < best_level:
            best, best_level = frame, level
    return best, False, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

def _decode_audio(audio_vae, audio_latent):
    out = audio_vae.decode(audio_latent)
    if isinstance(out, dict):
        out = out.get("waveform")
    return out


def _segment_noise(noise, seed_offset, latent):
    seed = int(getattr(noise, "seed", 0)) + seed_offset
    try:
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
        return Noise_RandomNoise(seed)
    except Exception:  # pragma: no cover - older ComfyUI
        return noise


class OnyxH3SeamlessSampler(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnyxH3SeamlessSampler",
            display_name="Onyx H3 Seamless Sampler",
            category="Onyx/Video",
            description=(
                "Long MiniMax H3 video with invisible seams: native latent extension (no <Video> reference), "
                "silence-aware handover so no word is ever cut, and a runtime timeline that plays every beat "
                "of the prompt exactly once. Works with any H3 prompt."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image", tooltip="Full-length empty MiniMax H3 AV latent (sets the total duration)."),
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Int.Input("segment_frames", default=175, min=39, max=3600, step=17,
                             tooltip="Frames generated per pass (VRAM). Snapped down to H3's 17k+5 grid."),
                io.Int.Input("context_frames", default=22, min=5, max=1000, step=17,
                             tooltip="Frames of the previous segment pinned as the opening of the next (5, 22, 39...)."),
                io.Float.Input("handover_search_seconds", default=6.0, min=0.0, max=20.0, step=0.1,
                               tooltip="How far back from the end of a segment the handover may move to find silence."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="H3 audio VAE. Enables the silence-aware handover (recommended)."),
                *[io.Image.Input(f"ref_image_{n}", optional=True,
                                 tooltip=f"Ref2VA reference image {n}, same as on MiniMax H3 Reference To Video.")
                  for n in range(1, 6)],
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.String.Output(display_name="segment_prompts"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, clip, prompt, fps=24.0,
                segment_frames=175, context_frames=22, handover_search_seconds=6.0, audio_vae=None,
                ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None, ref_image_5=None):
        _ensure_profile_ready()
        samples = latent_image["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError(f"{LOG} needs a MiniMax H3 nested video+audio latent")
        video, audio = samples.unbind()
        total_tokens = int(video.shape[2])
        total_frames = _pixel_frames(total_tokens)
        width, height = int(video.shape[4]) * 16, int(video.shape[3]) * 16

        max_seg = int(segment_frames) - (int(segment_frames) - 5) % 17
        guide = int(context_frames) - (int(context_frames) - 5) % 17
        if guide < 5 or guide + 17 > max_seg:
            raise ValueError(f"{LOG} context_frames must be 5, 22, 39... and at least 17 frames below segment_frames")
        search_frames = int(round(float(handover_search_seconds) * fps))
        images = [r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5) if r is not None] or None

        original_conds = guider.original_conds
        positive = original_conds.get("positive")
        if positive is None:
            raise ValueError(f"{LOG} needs a guider with positive conditioning")

        # A line must fit between the pinned guide and the next guide window.
        max_line = max_seg - 2 * guide - DEFER_OFFSET_FRAMES - SPEECH_MARGIN_FRAMES
        timeline = Timeline.parse(prompt, total_frames, fps, max_line_frames=max_line)
        multi = total_frames > max_seg
        kept_video, kept_audio, log_lines = [], [], []
        output_template = None
        produced = 0                 # global frames already delivered
        prev_video = prev_audio = None
        prev_cut = 0
        index = 0
        print(f"{LOG} {total_frames} frames, segments of {max_seg}, guide {guide}, "
              f"silence handover {'ON' if audio_vae is not None else 'OFF (no audio_vae)'}")
        try:
            while produced < total_frames:
                if index > 400:
                    raise RuntimeError(f"{LOG} no progress, aborting")
                first = index == 0
                g = 0 if first else guide
                remaining = total_frames - produced
                seg_frames = min(max_seg, g + remaining)
                is_last = seg_frames == g + remaining
                seg_start = produced - g          # global frame of local frame 0
                seg_tokens = _video_steps(seg_frames)
                seg_audio = _audio_steps(seg_frames)

                chosen = timeline.schedule(seg_start, seg_frames, g, 0 if is_last else guide, is_last)
                seg_prompt = timeline.segment_prompt(chosen, first, multi)
                header = (f"=== Segment {index + 1}: global frames {seg_start}-{seg_start + seg_frames - 1} "
                          f"(new from {produced}) ===")
                log_lines.append(header + "\n" + seg_prompt)
                print(f"{LOG} {header} — {len(chosen)} beat(s)")

                encoded = _encode_prompt(clip, seg_prompt, images, positive, width, height, not first)
                comfy.model_management.unload_model_and_clones(clip.patcher)

                video_context = audio_context = None
                if not first:
                    t_cut = _video_steps(prev_cut)
                    t_g = _video_steps(g)
                    video_context = prev_video[:, :, t_cut - t_g:t_cut].clone()
                    a_end = _audio_steps(prev_cut)
                    a_start = _audio_steps(prev_cut - g)
                    audio_context = prev_audio[..., a_start:a_end].clone()
                guider.original_conds = _conditioning_for_chunk(
                    original_conds, seg_start, seg_start + seg_frames, encoded,
                    video_context, audio_context, float(g), (), 0,
                )
                seg_video = torch.zeros((*video.shape[:2], seg_tokens, *video.shape[3:]),
                                        dtype=video.dtype, device=video.device)
                seg_audio_lat = torch.zeros((*audio.shape[:-1], seg_audio), dtype=audio.dtype, device=audio.device)
                seg_latent = latent_image.copy()
                seg_latent["samples"] = comfy.nested_tensor.NestedTensor((seg_video, seg_audio_lat))
                seg_noise = _segment_noise(noise, index, seg_latent)

                sampled, _denoised = super().execute(seg_noise, guider, sampler, sigmas, seg_latent)
                output_template = sampled.copy()
                output_template.pop("samples", None)
                out_video, out_audio = sampled["samples"].unbind()

                rms = threshold = None
                if is_last:
                    cut, silent = seg_frames, True
                else:
                    if audio_vae is not None:
                        try:
                            rms = frame_rms(_decode_audio(audio_vae, out_audio), seg_frames)
                        except Exception as error:  # never block a render on analysis
                            logging.warning("%s audio analysis failed, fixed handover: %s", LOG, error)
                    if timeline.rescheduled >= 2:
                        # Same line missed twice: stop moving the handover back,
                        # progress matters more than one perfect seam.
                        cut, silent, threshold = choose_handover(None, seg_frames, g, guide, 0)
                        rms = None
                    else:
                        cut, silent, threshold = choose_handover(rms, seg_frames, g, guide, search_frames)
                done = timeline.commit(chosen, cut, rms, threshold, 0 if is_last else guide)
                print(f"{LOG} segment {index + 1}: handover at local frame {cut}/{seg_frames} "
                      f"({'silent' if silent else 'quietest available'}), {done}/{len(chosen)} beat(s) done")

                t_cut = _video_steps(cut)
                t_g = _video_steps(g) if g else 0
                a_cut = _audio_steps(cut)
                a_g = _audio_steps(g) if g else 0
                kept_video.append(out_video[:, :, t_g:t_cut].detach().to("cpu"))
                kept_audio.append(out_audio[..., a_g:a_cut].detach().to("cpu"))
                produced += cut - g
                prev_video, prev_audio, prev_cut = out_video, out_audio, cut
                index += 1
        finally:
            guider.original_conds = original_conds

        final_video = torch.cat(kept_video, dim=2).to(video.device)
        final_audio = torch.cat(kept_audio, dim=-1).to(audio.device)
        target_a = audio.shape[-1]
        if final_audio.shape[-1] > target_a:
            final_audio = final_audio[..., :target_a]
        elif final_audio.shape[-1] < target_a:
            final_audio = torch.nn.functional.pad(final_audio, (0, target_a - final_audio.shape[-1]))
        output_template["samples"] = comfy.nested_tensor.NestedTensor((final_video, final_audio))
        print(f"{LOG} done: {index} segment(s), {produced} frames")
        return io.NodeOutput(output_template, "\n\n".join(log_lines))


__all__ = ["OnyxH3SeamlessSampler"]
