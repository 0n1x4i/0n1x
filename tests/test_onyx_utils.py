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
    return sys.modules[f"{pkg}.{name}"]


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
