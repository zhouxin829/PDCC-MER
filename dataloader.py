import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch


class MMDataset(Dataset):
    """Multimodal dataset with deterministic condition-specific degradation.

    The degradation is sample-static: every access to the same
    (split, index, condition, robust_seed) returns exactly the same input.
    This avoids an uncontrolled change of the corruption distribution across
    epochs and keeps all model random seeds on an identical corrupted dataset.
    """

    def __init__(self, args, mode='train', split_mode=''):
        self.mode = mode
        self.split_mode = split_mode
        self.args = args
        self.robust_mode = getattr(args, 'robust_mode', 'clean')
        self.robust_modality = getattr(args, 'robust_modality', 'none')
        self.robust_level = float(getattr(args, 'robust_level', 0.0))
        self.robust_scope = getattr(args, 'robust_scope', 'all')
        self.robust_seed = int(getattr(args, 'robust_seed', 20260707))
        self._validate_robust_config()

        data_map = {
            'SIMS': self.__init_sims,
            'SIMS-v2': self.__init_simsv2,
            'MOSI': self.__init_mosi,
            'MOSEI': self.__init_mosei,
        }
        data_map[args.dataset]()

    def _validate_robust_config(self):
        if self.robust_mode not in {
            'clean', 'missing', 'noise', 'random_missing', 'misalign'
        }:
            raise ValueError(f'Unknown robust_mode: {self.robust_mode}')
        if self.robust_modality not in {'none', 'text', 'audio', 'vision'}:
            raise ValueError(f'Unknown robust_modality: {self.robust_modality}')
        if self.robust_scope not in {'all', 'test_only'}:
            raise ValueError(f'Unknown robust_scope: {self.robust_scope}')
        if self.robust_mode == 'clean':
            return
        if self.robust_modality == 'none':
            raise ValueError(
                'robust_modality cannot be none when robust_mode is not clean'
            )
        if self.robust_mode in {'noise', 'random_missing', 'misalign'} and not (
            0.0 < self.robust_level <= 1.0
        ):
            raise ValueError(
                'robust_level must be in (0,1] for noise, random_missing, and misalign'
            )

    def __init_sims(self):
        if self.args.dataset == 'SIMS':
            if self.split_mode != '':
                path = os.path.join(
                    self.args.data_path, 'CH-' + self.args.dataset, 'Processed',
                    f'[{self.mode}]{self.split_mode}.pkl'
                )
            else:
                path = os.path.join(
                    self.args.data_path, 'CH-' + self.args.dataset, 'Processed',
                    'unaligned_39.pkl'
                )
        elif self.args.dataset == 'SIMS-v2':
            if self.split_mode != '':
                path = os.path.join(
                    self.args.data_path, 'CH-' + self.args.dataset, 'CH-SIMS-v2(s)',
                    'Processed', f'[{self.mode}]{self.split_mode}.pkl'
                )
            else:
                path = os.path.join(
                    self.args.data_path, 'CH-' + self.args.dataset, 'CH-SIMS-v2(s)',
                    'Processed', 'unaligned.pkl'
                )
        elif self.args.dataset in {'MOSI', 'MOSEI'}:
            path = os.path.join(
                self.args.data_path, 'CMU-' + self.args.dataset, 'Processed', 'unaligned_50.pkl'
            )
        else:
            raise ValueError(f'Unsupported dataset: {self.args.dataset}')

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.regression_m = data[self.mode]['regression_labels'].astype(np.float32)

        if 'SIMS' in self.args.dataset:
            classification_labels = {
                'M': np.where(data[self.mode]['regression_labels'] < 0, 0,
                              np.where(data[self.mode]['regression_labels'] == 0, 1, 2)),
                'T': np.where(data[self.mode]['regression_labels_T'] < 0, 0,
                              np.where(data[self.mode]['regression_labels_T'] == 0, 1, 2)),
                'A': np.where(data[self.mode]['regression_labels_A'] < 0, 0,
                              np.where(data[self.mode]['regression_labels_A'] == 0, 1, 2)),
                'V': np.where(data[self.mode]['regression_labels_V'] < 0, 0,
                              np.where(data[self.mode]['regression_labels_V'] == 0, 1, 2)),
            }
        else:
            common = np.where(data[self.mode]['regression_labels'] < 0, 0,
                              np.where(data[self.mode]['regression_labels'] == 0, 1, 2))
            classification_labels = {'M': common, 'T': common, 'A': common, 'V': common}

        self.labels = {key: value.astype(np.float32) for key, value in classification_labels.items()}
        self.text_lengths = np.sum(self.text[:, 1], axis=1).astype(np.int16).tolist()
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

    def __init_simsv2(self):
        return self.__init_sims()

    def __init_mosi(self):
        return self.__init_sims()

    def __init_mosei(self):
        return self.__init_sims()

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return (768, self.audio.shape[2], self.vision.shape[2])

    def _apply_to_this_split(self):
        if self.robust_mode == 'clean':
            return False
        if self.robust_scope == 'all':
            return True
        # Both default test and SIMS D_msc / D_msi use mode='test'.
        return self.mode == 'test'

    def _rng(self, index, salt):
        mode_code = {'train': 11, 'valid': 17, 'test': 23}.get(self.mode, 29)
        split_code = {'': 0, 'D_msc': 37, 'D_msi': 41}.get(self.split_mode, 43)
        seed = (
            int(self.robust_seed) + 1000003 * int(index) + 7919 * mode_code +
            1543 * split_code + 97 * int(salt)
        ) % (2**32 - 1)
        return np.random.default_rng(seed)

    @staticmethod
    def _valid_len(length, max_len):
        return max(0, min(int(length), int(max_len)))

    def _missing_text(self, text):
        return np.zeros_like(text, dtype=np.float32)

    def _missing_feature(self, feature):
        return np.zeros_like(feature, dtype=np.float32)

    def _noise_text(self, text, index):
        out = text.copy()
        if out.shape[0] < 2:
            return out
        token_ids = out[0]
        attention = out[1] > 0
        positions = np.arange(token_ids.shape[0])
        lexical = attention & (positions > 0) & (token_ids != 101) & (token_ids != 102)
        replace = (self._rng(index, 101).random(token_ids.shape[0]) < self.robust_level) & lexical
        # BERT [UNK] token ID is 100 for the BERT tokenizers used by this project.
        token_ids[replace] = 100.0
        out[0] = token_ids
        return out.astype(np.float32, copy=False)

    def _noise_feature(self, feature, length, index, salt):
        out = feature.copy().astype(np.float32, copy=False)
        valid_len = self._valid_len(length, out.shape[0])
        if valid_len == 0:
            return out
        valid = out[:valid_len]
        rms = float(np.sqrt(np.mean(np.square(valid), dtype=np.float64)))
        if not np.isfinite(rms) or rms <= 1e-12:
            return out
        noise = self._rng(index, salt).standard_normal(valid.shape).astype(np.float32)
        out[:valid_len] = valid + (self.robust_level * rms) * noise
        return out

    def _random_condition_applies(self, index, salt):
        return bool(self._rng(index, salt).random() < self.robust_level)

    def _misaligned_source_index(self, index):
        if len(self) <= 1:
            return -1
        # Draw from [0, n-2], then skip the current sample. This guarantees
        # that a corrupted modality always comes from a different utterance.
        donor = int(self._rng(index, 503).integers(0, len(self) - 1))
        if donor >= index:
            donor += 1
        return donor

    def _misalign(self, text, audio, vision, index):
        donor = self._misaligned_source_index(index)
        if donor < 0:
            return text, audio, vision, donor
        if self.robust_modality == 'text':
            text = self.text[donor].copy()
        elif self.robust_modality == 'audio':
            audio = self.audio[donor].copy()
        elif self.robust_modality == 'vision':
            vision = self.vision[donor].copy()
        return text, audio, vision, donor

    def _apply_robust_condition(self, text, audio, vision, index):
        if not self._apply_to_this_split():
            return text, audio, vision, False, -1

        modality = self.robust_modality
        applied = False
        donor = -1
        if self.robust_mode == 'missing':
            applied = True
            if modality == 'text':
                text = self._missing_text(text)
            elif modality == 'audio':
                audio = self._missing_feature(audio)
            elif modality == 'vision':
                vision = self._missing_feature(vision)
        elif self.robust_mode == 'noise':
            if modality == 'text':
                text = self._noise_text(text, index)
            elif modality == 'audio':
                audio = self._noise_feature(audio, self.audio_lengths[index], index, 211)
            elif modality == 'vision':
                vision = self._noise_feature(vision, self.vision_lengths[index], index, 307)
            applied = True
        elif self.robust_mode == 'random_missing':
            applied = self._random_condition_applies(index, 401)
            if applied:
                if modality == 'text':
                    text = self._missing_text(text)
                elif modality == 'audio':
                    audio = self._missing_feature(audio)
                elif modality == 'vision':
                    vision = self._missing_feature(vision)
        elif self.robust_mode == 'misalign':
            applied = self._random_condition_applies(index, 409)
            if applied:
                text, audio, vision, donor = self._misalign(
                    text, audio, vision, index
                )
                applied = donor >= 0
        return text, audio, vision, applied, donor

    def __getitem__(self, index):
        text = self.text[index].copy()
        audio = self.audio[index].copy()
        vision = self.vision[index].copy()
        text, audio, vision, robust_applied, robust_donor = (
            self._apply_robust_condition(text, audio, vision, index)
        )

        return {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(text),
            'text_lengths': self.text_lengths[index],
            'audio': torch.Tensor(audio),
            'audio_lengths': self.audio_lengths[index],
            'vision': torch.Tensor(vision),
            'vision_lengths': self.vision_lengths[index],
            'index': index,
            'id': self.ids[index],
            'robust_applied': robust_applied,
            'robust_donor_index': robust_donor,
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'regression_m': torch.tensor(self.regression_m[index]).view(-1),
        }
