import unittest

from ordinarylight import RenderPipeline, RenderStage


class RenderPipelineTests(unittest.TestCase):
    def test_validates_and_records_in_order(self):
        calls = []
        pipeline = RenderPipeline((
            RenderStage(
                "trace", reads={"scene"}, writes={"radiance"},
                recorder=lambda context: calls.append(("trace", context["frame"])),
            ),
            RenderStage(
                "denoise", reads={"radiance"}, writes={"filtered"},
                recorder=lambda context: calls.append(("denoise", context["frame"])),
            ),
            RenderStage(
                "present", reads={"filtered"}, writes={"swapchain"},
                recorder=lambda context: calls.append(("present", context["frame"])),
            ),
        ), initial_resources={"scene"})
        pipeline.record({"frame": 7})
        self.assertEqual(pipeline.stage_names, ("trace", "denoise", "present"))
        self.assertEqual(calls, [("trace", 7), ("denoise", 7), ("present", 7)])

    def test_rejects_missing_resource_and_duplicate_name(self):
        with self.assertRaisesRegex(ValueError, "unavailable resources"):
            RenderPipeline((RenderStage("bad", reads={"radiance"}),))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            RenderPipeline((
                RenderStage("same", writes={"a"}),
                RenderStage("same", reads={"a"}),
            ))

    def test_inserts_stage_without_mutating_pipeline(self):
        pipeline = RenderPipeline((
            RenderStage("trace", reads={"scene"}, writes={"radiance"}, recorder=lambda _: None),
            RenderStage("present", reads={"radiance"}, writes={"swapchain"}, recorder=lambda _: None),
        ), initial_resources={"scene"})
        result = pipeline.insert_before(
            "present",
            RenderStage("denoise", reads={"radiance"}, writes={"filtered"}, recorder=lambda _: None),
        )
        self.assertEqual(pipeline.stage_names, ("trace", "present"))
        self.assertEqual(result.stage_names, ("trace", "denoise", "present"))


if __name__ == "__main__":
    unittest.main()
