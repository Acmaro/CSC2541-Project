import pandas as pd
from tqdm import tqdm
from reinvent_chemistry import Conversions
from reinvent_chemistry.enums import FilterTypesEnum
from reinvent_chemistry.file_reader import FileReader
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry.standardization.filter_configuration import FilterConfiguration

from models.model import DecoratorModel
from models.rl_actions import SampleModel
from running_modes.configurations import ScaffoldDecoratingConfiguration
from running_modes.enums import GenerativeModelRegimeEnum


class ScaffoldDecorator:
    def __init__(self, configuration: ScaffoldDecoratingConfiguration, logger):
        self._configuration = configuration
        self._logger = logger
        self._model_regime = GenerativeModelRegimeEnum()
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._conversion = Conversions()

        filter_types = FilterTypesEnum()
        config = FilterConfiguration(name=filter_types.GET_LARGEST_FRAGMENT, parameters={})
        self._reader = FileReader([config], logger)

    def run(self):
        model = DecoratorModel.load_from_file(self._configuration.model_path, mode=self._model_regime.INFERENCE)
        input_scaffolds = list(
            self._reader.read_delimited_file(self._configuration.input_scaffold_path, standardize=True))

        input_scaffolds = [scaffold for scaffold in input_scaffolds if scaffold]
        input_scaffolds = input_scaffolds * self._configuration.number_of_decorations_per_scaffold

        sampling_action = SampleModel(model, self._configuration.batch_size, self._logger,
                                      self._configuration.randomize, sample_uniquely=self._configuration.sample_uniquely)
        sampled_sequences = sampling_action.run(input_scaffolds)

        # Collect rows in a list and build DataFrame once — avoids O(n²) pd.concat in loop
        rows = []
        n_invalid = 0
        for sample in tqdm(sampled_sequences, desc="Joining decorations", unit="mol", dynamic_ncols=True):
            scaffold = self._attachment_points.add_attachment_point_numbers(sample.scaffold, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, sample.decoration)

            if molecule:
                smile = self._conversion.mol_to_smiles(molecule, isomericSmiles=False, canonical=False)
                rows.append({
                    "SMILES":       smile,
                    "Scaffold":     sample.scaffold,
                    "Decorations":  sample.decoration,
                    "Likelihoods":  sample.nll,
                })
            else:
                n_invalid += 1
                self._logger.log_message(
                    f"Invalid decorations: {sample.decoration} for scaffold {sample.scaffold}")

        decorated_scaffolds = pd.DataFrame(rows, columns=["SMILES", "Scaffold", "Decorations", "Likelihoods"])

        self._logger.log_message(
            f"Sampled {len(decorated_scaffolds)} valid / {n_invalid} invalid scaffolds")
        decorated_scaffolds.to_csv(self._configuration.output_path, index=False)
