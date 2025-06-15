from absl.testing import absltest

from jax_trainer.logger.utils import flatten_dict


class TestLogger(absltest.TestCase):
    def test_flatten_configdict(self):
        config = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3,
                },
            },
            "f": {
                "g": 4,
            },
        }

        flattened_config = flatten_dict(config, separation_mark=".")
        self.assertEqual(flattened_config["a"], 1)
        self.assertEqual(flattened_config["b.c"], 2)
        self.assertEqual(flattened_config["b.d.e"], 3)
        self.assertEqual(flattened_config["f.g"], 4)
        self.assertEqual(len(flattened_config), 4)

        flatten_config = flatten_dict(config, separation_mark="/")
        self.assertEqual(flatten_config["a"], 1)
        self.assertEqual(flatten_config["b/c"], 2)
        self.assertEqual(flatten_config["b/d/e"], 3)
        self.assertEqual(flatten_config["f/g"], 4)
        self.assertEqual(len(flatten_config), 4)
