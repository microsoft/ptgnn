import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.tests.simplemodel.data import SampleDatapoint, SyntheticData
from ptgnn.tests.simplemodel.model import SimpleRegressionModel


class TestPytorchModel(unittest.TestCase):
    """
    Model of a Boolean classifier of positive/negative cross-product of input features and weights
    """

    def __get_data(
        self,
        num_points: int,
        num_features: int,
        random_seed: Optional[int] = None,
        train_test_split_pct=0.9,
    ) -> Tuple[List[SampleDatapoint], List[SampleDatapoint]]:
        """
        generates data as SyntheticData = SampleDataPoint[num_points] where
        SampleDataPoint<x,y> = <float[__num_features],bool>,  and y= sum(x*w) >0;
        weigths ~ N(0,1)*10, and SampleDataPoint.x ~ N(0,1)*5.
        Returns tuple of train and test data, split at @train_test_split_pct %
        """
        data = SyntheticData(num_features, random_seed=random_seed)
        all_data = list(data.generate(num_points))
        train_test_split = int(num_points * train_test_split_pct)
        training_data, validation_data = all_data[:train_test_split], all_data[train_test_split:]
        return training_data, validation_data

    def test_parallel(self):
        self.train_and_compare_model(True)

    def test_parallel_no_multiprocessing(self):
        self.train_and_compare_model(True, multiprocessing=False)

    def test_sequential(self):
        self.train_and_compare_model(False)

    def train_and_compare_model(self, parallelize: bool, multiprocessing: bool = True):
        num_points = 10000
        num_features = 100
        max_num_epochs = 50
        random_seed = 1234  # None to seed from clock
        training_data, validation_data = self.__get_data(
            num_points, num_features, random_seed=random_seed
        )

        with tempfile.TemporaryDirectory() as dir:
            model_file = Path(dir) / "testModel.pkl.gz"

            model = SimpleRegressionModel()
            trainer = ModelTrainer(model, model_file, max_num_epochs=max_num_epochs)
            trainer.train(
                training_data,
                validation_data,
                parallelize=parallelize,
                use_multiprocessing=multiprocessing,
            )
            model_acc_1 = model.compute_accuracy(
                trainer.neural_module,
                iter(validation_data),
                parallelize,
                use_multiprocessing=multiprocessing,
            )

            model, trained_network = SimpleRegressionModel.restore_model(model_file)
            trained_model_acc = model.compute_accuracy(
                trained_network,
                iter(validation_data),
                parallelize,
                use_multiprocessing=multiprocessing,
            )
            self.assertGreater(
                trained_model_acc, 0.95, f"Model achieves too low accuracy, {trained_model_acc:%}"
            )

            self.assertAlmostEqual(
                trained_model_acc,
                model_acc_1,
                places=3,
                msg=f"Accuracy before and after loading does not match: {trained_model_acc} vs {model_acc_1}",
            )


if __name__ == "__main__":
    unittest.main()
