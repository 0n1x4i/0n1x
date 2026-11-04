# -*- coding: utf-8 -*-
"""Tests for the shared utils (no ComfyUI, no network, no torch needed).

    python -m unittest discover -s tests -v      # from the pack root
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["ONYX_REGISTRY_URL"] = "off"

import requests  # noqa: E402

from utils import onyx_http, onyx_registry, onyx_cost, onyx_parallel, onyx_api_cache, onyx_opencv  # noqa: E402


class FakeResp:
    def __init__(self, status, body=None, headers=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")
        self.content = self.text.encode()

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def request(self, *a, **k):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class HttpTests(unittest.TestCase):
    """Each request is sent once: no retry on any error."""

    def test_429_is_not_retried(self):
        s = FakeSession([FakeResp(429, {"error": {"message": "slow down"}}), FakeResp(200, {"ok": 1})])
        with self.assertRaises(onyx_http.OnyxAPIError) as cm:
            onyx_http.post_json("https://x", service="T", payload={}, session=s)
        self.assertEqual(s.calls, 1)
        self.assertEqual(cm.exception.status, 429)

    def test_5xx_is_not_retried(self):
        s = FakeSession([FakeResp(503, {"message": "busy"}), FakeResp(200, {})])
        with self.assertRaises(onyx_http.OnyxAPIError) as cm:
            onyx_http.post_json("https://x", service="T", payload={}, session=s)
        self.assertEqual(s.calls, 1)
        self.assertIn("busy", str(cm.exception))

    def test_network_errors_not_retried(self):
        for exc in (requests.exceptions.ConnectionError("dns"), requests.exceptions.ReadTimeout("slow")):
            s = FakeSession([exc, FakeResp(200, {})])
            with self.assertRaises(onyx_http.OnyxAPIError):
                onyx_http.post_json("https://x", service="T", payload={}, session=s)
            self.assertEqual(s.calls, 1)
            s = FakeSession([exc, FakeResp(200, {})])
            with self.assertRaises(onyx_http.OnyxAPIError):
                onyx_http.get_json("https://x", service="T", session=s)
            self.assertEqual(s.calls, 1)

    def test_success(self):
        s = FakeSession([FakeResp(200, {"ok": 1})])
        self.assertEqual(onyx_http.post_json("https://x", service="T", payload={}, session=s), {"ok": 1})

    def test_400_message(self):
        s = FakeSession([FakeResp(400, {"error": {"message": "bad size", "status": "INVALID_ARGUMENT"}})])
        with self.assertRaises(onyx_http.OnyxAPIError) as cm:
            onyx_http.post_json("https://x", service="T", payload={}, session=s)
        self.assertIn("bad size", str(cm.exception))

    def test_key_is_redacted(self):
        key = "AIzaSySECRETSECRET123"
        s = FakeSession([FakeResp(403, {"error": {"message": f"key {key} not allowed"}})])
        with self.assertRaises(onyx_http.OnyxAPIError) as cm:
            onyx_http.post_json("https://x", service="T", payload={}, session=s,
                                headers={"x-goog-api-key": key})
        self.assertNotIn(key, str(cm.exception))
        self.assertIn("?key=***", onyx_http.redact("https://g/v1?key=abcdef123456&x=1"))

    def test_fal_detail_shape(self):
        s = FakeSession([FakeResp(422, {"detail": [{"loc": ["body", "image_size"], "msg": "too big"}]})])
        with self.assertRaises(onyx_http.OnyxAPIError) as cm:
            onyx_http.post_json("https://x", service="fal", payload={}, session=s)
        self.assertIn("image_size: too big", str(cm.exception))


class RegistryTests(unittest.TestCase):
    def test_deprecated_sorted_last_and_kept(self):
        ids = onyx_registry.model_ids("grok")
        self.assertIn("grok-2-vision-1212", ids)
        current = [i for i in ids if not onyx_registry.entry("grok", i).get("deprecated")]
        self.assertEqual(ids[:len(current)], current)

    def test_filters_and_always(self):
        ids = onyx_registry.model_ids("grok", vision=True, include_deprecated=False, always=("old-x",))
        self.assertNotIn("grok-3", ids)
        self.assertNotIn("grok-2-vision-1212", ids)
        self.assertEqual(ids[-1], "old-x")

    def test_merge_adds_and_updates_never_removes(self):
        base = {"g": {"default": "a", "entries": [{"id": "a"}, {"id": "b"}]}}
        onyx_registry._merge_models(base, {"g": {"entries": [{"id": "b", "deprecated": True}, {"id": "c"}]}})
        self.assertEqual([e["id"] for e in base["g"]["entries"]], ["a", "b", "c"])
        self.assertTrue(base["g"]["entries"][1]["deprecated"])

    def test_capabilities(self):
        self.assertIn("gemini-3.6-flash", onyx_registry.no_sampling_models())
        self.assertNotIn("gemini-3.5-flash", onyx_registry.no_sampling_models())
        self.assertNotIn("minimal", onyx_registry.thinking_support()["gemini-2.5-pro"])


class CostTests(unittest.TestCase):
    def test_exact_image_price(self):
        e = onyx_cost.estimate_aio({"provider": "GOOGLE", "model": "Nano Banana Pro",
                                    "image_size": "4K", "batch_size": 3})
        self.assertEqual(e.confidence, "exact")
        self.assertAlmostEqual(e.total, 0.72)

    def test_indicative_for_reseller(self):
        e = onyx_cost.estimate_aio({"provider": "KIE", "model": "Nano Banana 2",
                                    "image_size": "1K", "batch_size": 1})
        self.assertEqual(e.confidence, "indicative")
        self.assertIn("≈", e.label())

    def test_unknown_is_not_invented(self):
        e = onyx_cost.estimate_aio({"provider": "KIE", "model": "Seedream 5 Pro", "image_size": "2K"})
        self.assertIsNone(e.total)
        self.assertIn("unknown", e.label())

    def test_video_per_second(self):
        e = onyx_cost.estimate_aio({"provider": "GOOGLE", "model": "Veo 3.1 Fast", "image_size": "1080p",
                                    "video_duration": 8, "video_mode_enabled": True})
        self.assertAlmostEqual(e.total, 0.96)


class ParallelTests(unittest.TestCase):
    def test_order_preserved(self):
        def f(x):
            time.sleep(0.01 * (5 - x))
            return x * 10
        self.assertEqual(onyx_parallel.map_parallel(f, range(5), max_workers=5, progress=False),
                         [0, 10, 20, 30, 40])

    def test_first_error_raised(self):
        def f(x):
            if x == 2:
                raise ValueError("boom")
            return x
        with self.assertRaises(ValueError):
            onyx_parallel.map_parallel(f, range(5), max_workers=3, progress=False)

    def test_return_exceptions(self):
        out = onyx_parallel.map_parallel(lambda x: 1 // x, [1, 0, 2], max_workers=2,
                                         return_exceptions=True, progress=False)
        self.assertEqual(out[0], 1)
        self.assertIsInstance(out[1], ZeroDivisionError)


class CacheKeyTests(unittest.TestCase):
    def test_key_ignores_api_keys_and_is_stable(self):
        a = onyx_api_cache.make_key("n", {"prompt": "x", "gemini_api_key": "k1", "seed": 1})
        b = onyx_api_cache.make_key("n", {"seed": 1, "gemini_api_key": "k2", "prompt": "x"})
        self.assertEqual(a, b)

    def test_key_changes_with_params_and_pixels(self):
        import numpy as np
        img = np.zeros((4, 4, 3), dtype=np.float32)
        k1 = onyx_api_cache.make_key("n", {"prompt": "x", "image_1": img})
        img2 = img.copy()
        img2[0, 0, 0] = 1.0
        self.assertNotEqual(k1, onyx_api_cache.make_key("n", {"prompt": "x", "image_1": img2}))
        self.assertNotEqual(k1, onyx_api_cache.make_key("n", {"prompt": "y", "image_1": img}))
        self.assertNotEqual(k1, onyx_api_cache.make_key("other", {"prompt": "x", "image_1": img}))

    def test_unknown_object_disables_cache(self):
        class Weird:
            pass
        self.assertIsNone(onyx_api_cache.make_key("n", {"x": Weird()}))

    def test_store_and_load_texts_and_video(self):
        tmp = tempfile.mkdtemp()
        try:
            with mock.patch.object(onyx_api_cache, "CACHE_DIR", tmp):
                video = os.path.join(tmp, "v.mp4")
                with open(video, "wb") as fh:
                    fh.write(b"\0" * 100)
                onyx_api_cache.store("k", video_path=video, texts=["", "sources"])
                hit = onyx_api_cache.load("k")
                self.assertEqual(hit["texts"], ["", "sources"])
                self.assertTrue(os.path.isfile(hit["video"]))
                self.assertIsNone(onyx_api_cache.load("missing"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class OpenCVPlanTests(unittest.TestCase):
    def test_no_conflict(self):
        self.assertEqual(onyx_opencv.plan({"opencv-python-headless": "4.11.0"})["uninstall"], [])

    def test_keeps_contrib_and_newest_version(self):
        p = onyx_opencv.plan({"opencv-python": "4.10.0.84", "opencv-contrib-python": "4.11.0.86"})
        self.assertEqual(p["install"], "opencv-contrib-python-headless==4.11.0.86")
        self.assertEqual(len(p["uninstall"]), 2)

    def test_headless_when_no_contrib(self):
        p = onyx_opencv.plan({"opencv-python": "4.9.0.80", "opencv-python-headless": "4.10.0.84"})
        self.assertEqual(p["install"], "opencv-python-headless==4.10.0.84")


if __name__ == "__main__":
    unittest.main()


def _load_pack_module(name):
    """Import one pack module through the doc generator's ComfyUI stubs."""
    import runpy
    gen = runpy.run_path(os.path.join(ROOT, "tools", "gen_nodes_doc.py"), run_name="gen_for_tests")
    pkg = gen["PKG"]
    if pkg not in sys.modules:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            gen["load_pack"]()
        _offline_worker(pkg)
    return sys.modules[f"{pkg}.{name}"]


def _offline_worker(pkg):
    """Serve the protected cores/bundles from license-worker/protected_nodes (dev checkout
    only) instead of the Worker, so the moved logic is tested too."""
    import hashlib
    protected = os.path.join(ROOT, "license-worker", "protected_nodes")
    rex = sys.modules.get(f"{pkg}.nodes.onyx_remote_exec")
    if rex is None or not os.path.isdir(protected):
        return

    class _Resp:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    class _Requests:
        @staticmethod
        def post(url, json=None, **kw):
            path = os.path.join(protected, (json or {}).get("name", "") + ".py")
            if not os.path.isfile(path):
                return _Resp({"valid": False, "reason": "not found"})
            src = open(path, encoding="utf-8").read().replace("{{RID}}", "test")
            return _Resp({"valid": True, "source": src, "hash": hashlib.sha256(src.encode()).hexdigest()[:16]})

    rex.requests = _Requests
    rex._read_cache = lambda: {"token": "test"}
    rex._ensure_synced = lambda token: None
    rex._get_node_id = lambda: "test"
    rd = sys.modules.get(f"{pkg}.utils.onyx_remote_data")
    if rd is not None:
        rd._state.update(data={}, at=0.0, failed_at=0.0)
        th = rd.refresh(force=True)
        if th is not None:
            th.join(30)


class H3VfxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vfx = _load_pack_module("nodes.onyx_h3_vfx")

    def test_trigger_and_source_tags(self):
        v = self.vfx
        mode = v.resolve(v.VFX_EDIT)
        out = v.apply_prompt_rules("Make the jacket in <Video 1> red, like <Video 2>.", mode,
                                   source_audio_guided=False)
        self.assertEqual(out, "vfx_edit: Make the jacket in the source video red, like <Video 1>.")

    def test_trigger_not_doubled_and_audio(self):
        v = self.vfx
        mode = v.resolve(v.VFX_STYLE)
        out = v.apply_prompt_rules("style_transfer: keep <Audio 1>", mode, source_audio_guided=True)
        self.assertEqual(out, "style_transfer: keep the source soundtrack")

    def test_required_reference(self):
        v = self.vfx
        with self.assertRaises(RuntimeError):
            v.check_inputs(v.resolve(v.VFX_EDIT_FRAME), has_source_video=True, image_count=0, node="T")
        with self.assertRaises(RuntimeError):
            v.check_inputs(v.resolve(v.VFX_EDIT), has_source_video=False, image_count=0, node="T")

    def test_postprocess_shapes(self):
        v = self.vfx
        mode = v.resolve(v.VFX_EDIT)
        self.assertEqual(v.postprocess_result(("a <Video 1>", "{}"), mode, source_audio_guided=False,
                                              has_audio_input=False, node="T"),
                         ("vfx_edit: a the source video", "{}"))
        wrapped = v.postprocess_result({"ui": {}, "result": ("x", "{}")}, None,
                                       source_audio_guided=False, has_audio_input=True, node="T")
        self.assertEqual(wrapped["result"], ("x", "{}"))


class SetupCheckTests(unittest.TestCase):
    def test_missing_nodes_models_and_renamed(self):
        from utils import onyx_setup_check as sc
        fake_nodes = mock.MagicMock(NODE_CLASS_MAPPINGS={"KSampler": 1, "UNETLoader": 1})
        fp = mock.MagicMock(folder_names_and_paths={"loras": 1, "vae": 1})
        fp.get_filename_list = lambda folder: {"loras": ["sub/my_lora.safetensors"], "vae": ["ok.safetensors"]}[folder]
        with mock.patch.dict(sys.modules, {"nodes": fake_nodes, "folder_paths": fp}):
            sc._cache.update(at=0.0, files=None)
            r = sc.check(["KSampler", "PathchSageAttentionKJ"], [
                {"node_type": "VAELoader", "widget": "vae_name", "value": "ok.safetensors"},
                {"node_type": "LoraLoaderModelOnly", "widget": "lora_name", "value": "my_lora.safetensors"},
                {"node_type": "UNETLoader", "widget": "unet_name", "value": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"},
                {"node_type": "LoadImage", "widget": "image", "value": "photo.png"},
            ])
        self.assertFalse(r["ok"])
        self.assertEqual(r["missing_nodes"][0]["pack"], "ComfyUI-KJNodes")
        self.assertEqual(len(r["missing_models"]), 1)
        self.assertEqual(r["missing_models"][0]["folder"], "diffusion_models")
        self.assertTrue(r["missing_models"][0]["url"].startswith("https://huggingface.co/"))
        self.assertEqual(r["elsewhere"][0]["found_as"], "sub/my_lora.safetensors")


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = _load_pack_module("nodes.onyx_studio")
        cls.llm = _load_pack_module("utils.onyx_llm")

    def test_parse_answer(self):
        r = self.llm.parse_answer("READY\n```\nmia woman\nA shot.\n```\nNOTES: ratio 4:5 · post: shake_med_h")
        self.assertEqual(r["prompt"], "mia woman\nA shot.")
        self.assertEqual((r["ratio"], r["post"]), ("4:5", "shake_med_h"))
        r = self.llm.parse_answer("<think>hmm</think>plain prompt text")
        self.assertEqual((r["prompt"], r["post"]), ("plain prompt text", ""))

    def test_render_skill_swaps_character(self):
        text = ("TRIGGER_LINE = style_x, anna woman\nBODY = a petite woman\nOUTPUT = raw\n"
                'S2 HAIR LOCK "Her hair is a short dark cut that sits\n'
                '   just above her collar."\nExample:\nstyle_x, anna woman\nA candid snapshot of a petite woman.')
        meta = {"locked_sentence": "Her hair is a short dark cut that sits just above her collar."}
        char = {"trigger_line": "mia woman", "body": "a slim woman", "hair_lock": "Her long red hair."}
        out = self.st.render_skill(text, meta, char, "4:5")
        self.assertNotIn("anna", out)
        self.assertNotIn("petite", out)
        self.assertIn("Her long red hair.", out)
        self.assertIn("OUTPUT = chat", out)
        self.assertNotIn("short dark cut", out)

    def test_trigger_handling(self):
        char = {"trigger_line": "mia woman"}
        self.assertEqual(self.st._apply_trigger("{trigger}, on a bench", char, True), "mia woman, on a bench")
        self.assertEqual(self.st._apply_trigger("on a bench", char, True), "mia woman\non a bench")
        self.assertEqual(self.st._apply_trigger("anna woman\nA shot.", char, True, from_llm=True), "mia woman\nA shot.")
        self.assertEqual(self.st._apply_trigger("on a bench", char, False), "on a bench")


class PostLookTests(unittest.TestCase):
    def setUp(self):
        self.p = _load_pack_module("utils.onyx_post")

    def test_none_changes_nothing(self):
        post = self.p.make("none", "none")
        kw = {"exposure": 0.5}
        self.assertEqual(self.p.apply("OnyxPostProcessSuite", kw, post), (kw, False))

    def test_look_overrides_only_existing_widgets(self):
        post = self.p.make("night", "none")
        out, bypass = self.p.apply("Onyx_Renoise", {"iso_preset": "ISO 100 (Clean)", "seed": 3}, post)
        self.assertFalse(bypass)
        self.assertEqual(out["iso_preset"], "ISO 800")
        self.assertEqual(out["seed"], 3)
        self.assertNotIn("fine_grain", out)          # not a widget passed in -> never injected

    def test_bypass_and_motion(self):
        post = self.p.make("clean", "auto", auto_motion="shake_med_h")
        self.assertTrue(self.p.apply("Onyx_Camera_Look", {"enabled": True}, post)[1])
        self.assertEqual(post["nodes"]["OnyxCameraShake"]["preset"], "shake_med_h")
        self.assertEqual(self.p.make("none", "shake_heavy")["nodes"], {"OnyxCameraShake": {"preset": "shake_heavy"}})
        self.assertEqual(self.p.make("amateur", "none")["nodes"]["OnyxCameraShake"]["preset"], "none")

    def test_motion_none_switches_off_manual_keeps_node_settings(self):
        kw = {"preset": "shake_med", "strength": 1.0}
        off = self.p.make("none", "none")
        self.assertEqual(self.p.apply("OnyxCameraShake", kw, off)[0]["preset"], "none")
        manual = self.p.make("none", "manual")
        self.assertEqual(self.p.apply("OnyxCameraShake", kw, manual), (kw, False))
        self.assertNotIn("OnyxCameraShake", self.p.make("amateur", "manual")["nodes"])
        self.assertIn("manual", self.p.MOTIONS)

    def test_wire_adds_input_and_applies(self):
        calls = {}

        class Node:
            FUNCTION = "run"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",), "jpeg_compression": ("INT", {})}}

            def run(self, image, jpeg_compression):
                calls["q"] = jpeg_compression
                return (image,)

        self.p.wire(Node, "Onyx_Camera_Look")
        self.assertIn("onyx_post", Node.INPUT_TYPES()["optional"])
        Node().run(image="IMG", jpeg_compression=100, onyx_post=self.p.make("amateur", "none"))
        self.assertEqual(calls["q"], 92)
        self.assertEqual(Node().run(image="IMG", jpeg_compression=100, onyx_post=self.p.make("clean", "none")), ("IMG",))


class CameraShakeTableTests(unittest.TestCase):
    def test_presets_match_shake_py(self):
        mod = _load_pack_module("nodes.onyx_camera_shake")
        presets = mod._onyx_core()["PRESETS"]          # protected: lives in postfx_bundle
        self.assertEqual(presets["shake_med_rot"], (0.012, 60, 0.008, 0.20, 0.0008, 8))
        self.assertEqual(presets["soft"], (0.000, 0, 0.000, 0.00, 0.0012, 0))
        self.assertEqual(mod.PRESET_NAMES, ["none", *presets])   # local dropdown in sync


class StudioPresetResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = _load_pack_module("nodes.onyx_studio")

    def test_presets_round_trip(self):
        tmp = tempfile.mkdtemp()
        try:
            with mock.patch.object(self.st, "studio_dir", lambda: tmp):
                self.st.save_preset(3, "Night", {"mode": "idea", "tab": "post", "gallery_q": "x"},
                                    {"video_max_side": 768, "batch_data": "{...}"})
                p = self.st.presets()["3"]
                self.assertEqual(p["name"], "Night")
                self.assertNotIn("tab", p["state"])               # UI-only keys are not saved
                self.assertNotIn("batch_data", p["widgets"])      # the media list is not a setting
                with self.assertRaises(ValueError):
                    self.st.save_preset(11, "x", {}, {})
                self.st.delete_preset(3)
                self.assertEqual(self.st.presets(), {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolution_follows_skill_ratio(self):
        calls = {}

        class FakeRes:
            def compute(self, megapixels, aspect_ratio, multiple_of, image=None):
                calls.update(mp=megapixels, ar=aspect_ratio, mult=multiple_of)
                return (768, 1344, aspect_ratio)

        mod = sys.modules[self.st.__name__.rsplit(".", 1)[0] + ".onyx_resolution_mp"]
        with mock.patch.object(mod, "OnyxResolutionMP", FakeRes):
            w, h, r = self.st.resolution({"ratio": "9:16", "res": {"mp": 1.03, "mult": 32}}, "4:5")
            self.assertEqual((w, h, calls["ar"], calls["mp"]), (768, 1344, "9:16", 1.03))
            self.st.resolution({"ratio": "9:16", "res": {"follow_skill": True}}, "4:5")
            self.assertEqual(calls["ar"], "4:5")
            self.st.resolution({"ratio": "from image (input)", "res": {}}, "", image=None)
            self.assertEqual(calls["ar"], "1:1")                 # no image: square fallback


class StudioCacheKeyTests(unittest.TestCase):
    def test_editing_the_character_changes_the_key(self):
        st = _load_pack_module("nodes.onyx_studio")
        tmp = tempfile.mkdtemp()
        try:
            with mock.patch.object(st, "studio_dir", lambda: tmp):
                c = st.save_character({"name": "Mia", "trigger_line": "mia"})
                state = json.dumps({"mode": "manual", "manual": "x", "character_id": c["id"]})
                k1 = st.OnyxStudio.IS_CHANGED(state, 1)
                self.assertEqual(k1, st.OnyxStudio.IS_CHANGED(state, 1))
                st.save_character({**c, "trigger_line": "mia_v2"})
                self.assertNotEqual(k1, st.OnyxStudio.IS_CHANGED(state, 1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        self.cc = _load_pack_module("nodes.onyx_conditioning_cache")
        self.tmp = tempfile.mkdtemp()
        self._dir = mock.patch.object(self.cc, "CACHE_DIR", self.tmp)
        self._dir.start()
        # the logic runs in the protected core (its own namespace): patch it too
        self._core = mock.patch.dict(self.cc._onyx_core(), {"CACHE_DIR": self.tmp})
        self._core.start()
        self.cc._memory.clear()
        self.cc._onyx_core()["_memory"].clear()

    def tearDown(self):
        self._core.stop()
        self._dir.stop()
        self.cc._memory.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fingerprint_follows_the_encoder_chain(self):
        g = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors"}},
             "2": {"class_type": "OnyxConditioningCache", "inputs": {"text": "x", "clip": ["1", 0]}}}
        fp = self.cc.encoder_fingerprint(g, "2")
        self.assertTrue(fp)
        g["1"]["inputs"]["clip_name"] = "qwen_3_8b.safetensors"
        self.assertNotEqual(fp, self.cc.encoder_fingerprint(g, "2"))
        self.assertEqual(self.cc.encoder_fingerprint(None, "2"), "")

    def test_prune_deletes_least_recently_used_first(self):
        for i, name in enumerate(["old", "mid", "new"]):
            path = os.path.join(self.tmp, f"{name}.pt")
            with open(path, "wb") as f:
                f.write(b"x" * 1000)
            os.utime(path, (1000 + i, 1000 + i))
        self.assertEqual(self.cc.prune(2500), 1)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["mid.pt", "new.pt"])
        self.assertEqual(self.cc.stats()["files"], 2)
        self.assertEqual(self.cc.clear(), 2)


class StudioEncodeCacheTests(unittest.TestCase):
    def setUp(self):
        self.st = _load_pack_module("nodes.onyx_studio")

    def test_only_fixed_prompts_are_cacheable(self):
        st = self.st
        self.assertEqual(st._cacheable_prompt({"mode": "manual", "manual": "a red car"}), "a red car")
        self.assertIsNone(st._cacheable_prompt({"mode": "manual", "manual": "x", "cond_cache": False}))
        self.assertIsNone(st._cacheable_prompt({"mode": "manual", "manual": "x", "enhance": True, "skill": "s"}))
        self.assertIsNone(st._cacheable_prompt({"mode": "idea", "idea": "x", "skill": "s"}))
        self.assertIsNone(st._cacheable_prompt({"mode": "recreate"}))

    def test_encoder_is_requested_unless_cached_and_clip_output_unused(self):
        st = self.st
        g = {"5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5.safetensors"}},
             "9": {"class_type": "OnyxStudio", "inputs": {"clip": ["5", 0]}},
             "10": {"class_type": "KSampler", "inputs": {"positive": ["9", 2]}}}
        state = json.dumps({"mode": "manual", "manual": "a red car"})
        node = st.OnyxStudio()
        with mock.patch.object(st._cond_cache, "lookup", return_value=None):
            self.assertEqual(node.check_lazy_status(state, "9", g), ["clip"])        # not cached yet
        with mock.patch.object(st._cond_cache, "lookup", return_value=[["c", {}]]):
            self.assertEqual(node.check_lazy_status(state, "9", g), [])              # cached: no encoder
            g["11"] = {"class_type": "Other", "inputs": {"clip": ["9", 1]}}
            self.assertEqual(node.check_lazy_status(state, "9", g), ["clip"])        # clip output used
            del g["11"]
            idea = json.dumps({"mode": "idea", "idea": "x", "skill": "s"})
            self.assertEqual(node.check_lazy_status(idea, "9", g), ["clip"])         # prompt changes
            del g["9"]["inputs"]["clip"]
            self.assertEqual(node.check_lazy_status(state, "9", g), [])              # nothing to load


class PostStudioTests(unittest.TestCase):
    def setUp(self):
        self.ps = _load_pack_module("nodes.onyx_post_studio")

    def test_chain_order_is_the_crispy_workflow_with_motion_last(self):
        keys = self.ps.STAGE_KEYS
        crispy = [k for k in keys if k not in self.ps._DEFAULT_OFF]
        self.assertEqual(crispy, ["contrast", "color", "skin", "grade", "camera", "lens", "grain"])   # the default look
        self.assertEqual(keys[:3], ["curves", "levels", "contrast"])               # the light comes first
        self.assertEqual(keys[-1], "chroma")                        # the upload, after the photo exists
        self.assertLess(keys.index("lut"), keys.index("camera"))
        self.assertLess(keys.index("clip"), keys.index("camera"))
        self.assertLess(keys.index("camera"), keys.index("hdr"))
        self.assertLess(keys.index("hdr"), keys.index("grain"))

    def _fake(self, calls, name, extra=None):
        class Fake:
            FUNCTION = "go"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",), "amount": ("FLOAT", {"default": 1.0}),
                                     **(extra or {})}}

            def go(self, image, amount, onyx_post=None, **kw):
                calls.append((name, amount, kw))
                return (image + [name],)
        return Fake

    def test_runs_enabled_stages_in_order_with_values(self):
        ps, calls = self.ps, []
        classes = {k: self._fake(calls, k) for k in ps.STAGE_KEYS}
        with mock.patch.object(ps, "stage_class", lambda k: classes.get(k)), \
             mock.patch.object(ps, "defaults", lambda k, cls=None: {"amount": 1.0}), \
             mock.patch.dict(ps._onyx_core(), {"stage_class": lambda k: classes.get(k),
                                               "defaults": lambda k, cls=None: {"amount": 1.0}}):
            out, timings = ps.run_chain([], {"stages": {"grain": {"on": False}, "hdr": {"on": True},
                                                       "skin": {"v": {"amount": "0.5", "bogus": 3}}}})
        self.assertEqual(out, ["contrast", "color", "skin", "grade", "camera", "lens", "hdr"])
        self.assertEqual(calls[2], ("skin", 0.5, {}))               # coerced, unknown key dropped


    def test_preview_skips_a_failing_stage_but_a_run_raises(self):
        ps = self.ps

        class Boom:
            FUNCTION = "go"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)}}

            def go(self, image, **kw):
                raise RuntimeError("no licence")
        calls = []
        classes = {k: (Boom if k == "grain" else self._fake(calls, k)) for k in ps.STAGE_KEYS}
        classes = {k: v for k, v in classes.items() if k not in ps._DEFAULT_OFF}
        dflt = lambda k, cls=None: {"amount": 1.0} if k != "grain" else {}  # noqa: E731
        with mock.patch.object(ps, "stage_class", lambda k: classes.get(k)), \
             mock.patch.object(ps, "defaults", dflt), \
             mock.patch.dict(ps._onyx_core(), {"stage_class": lambda k: classes.get(k), "defaults": dflt}):
            out, timings = ps.run_chain([], {}, preview=True)
            self.assertNotIn("grain", out)
            self.assertIn(("grain", "no licence"), timings)
            with self.assertRaises(RuntimeError):
                ps.run_chain([], {})

    def test_widgets_are_grouped_by_enable_switches(self):
        ps = self.ps

        class Suite:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)},
                        "optional": {"enable_glow": ("BOOLEAN", {"default": False}),
                                     "glow_radius": ("FLOAT", {"default": 2.0, "min": 0, "max": 9}),
                                     "global_mix": ("FLOAT", {"default": 1.0}),
                                     "mode": (["a", "b"],), "enabled": ("BOOLEAN", {}),
                                     "hook": ("STRING", {"forceInput": True})}}
        params = {p["name"]: p for p in ps._params(Suite)}
        self.assertEqual(set(params), {"enable_glow", "glow_radius", "global_mix", "mode"})
        self.assertTrue(params["enable_glow"]["switch"])
        self.assertEqual(params["glow_radius"]["group"], "glow")
        self.assertEqual(params["global_mix"]["group"], "main")
        self.assertEqual(params["mode"]["default"], "a")


class PhoneFxTests(unittest.TestCase):
    def setUp(self):
        self.fx = _load_pack_module("nodes.onyx_phone_fx")
        import numpy as np
        yy, xx = np.mgrid[0:96, 0:128]
        self.img = np.stack([xx / 128, yy / 96, 0.3 + 0 * xx], -1).astype(np.float32)
        self.img[30:40, 50:60] = 0.98                                   # a lamp

    def test_hdr_lifts_shadows_and_holds_highlights(self):
        out = self.fx.phone_hdr(self.img, strength=1.0, sharpen_halo=0.0, local_contrast=0.0)
        dark = self.img[..., 0] + self.img[..., 1] < 0.3
        self.assertGreater(out[dark].mean(), self.img[dark].mean())
        self.assertLess(out[30:40, 50:60].mean(), self.img[30:40, 50:60].mean() + 1e-3)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_clip_burns_only_bright_areas(self):
        out = self.fx.highlight_clip(self.img, threshold=0.8, push=2.0, bloom=0.0, fringe=0.0)
        self.assertGreater(out[35, 55].min(), 0.99)                      # the lamp is white
        self.assertLess(abs(float(out[10, 5].mean() - self.img[10, 5].mean())), 0.005)

    def test_smudge_follows_the_light(self):
        import numpy as np
        dark = np.full((96, 128, 3), 0.1, np.float32)
        self.assertLess(float(np.abs(self.fx.lens_smudge(dark, haze=0.0) - dark).max()), 1e-6)
        out = self.fx.lens_smudge(self.img, amount=1.0)
        self.assertGreater(out[35, 70].mean(), self.img[35, 70].mean())  # glow next to the lamp

    def test_chroma_subsample_keeps_luma(self):
        import torch
        t = torch.rand(1, 33, 47, 3)
        out = self.fx.chroma_subsample(t, "4:2:0")
        y = lambda x: 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        self.assertEqual(out.shape, t.shape)
        self.assertLess(float((y(out) - y(t)).abs().mean()), 0.02)
        self.assertGreater(float((out - t).abs().mean()), 0.01)          # colour did change
