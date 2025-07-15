import torch
import os
import json
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class LungCTDataset(Dataset):
    def __init__(self, root_dir, task='complicacion', transform=None, use_monai_io=False, only_labels=False):
        """
        Dataset para volúmenes pulmonares NIfTI con nombres codificados.

        Args:
            root_dir (str): Carpeta con archivos .nii.gz.
            task (str): Tarea de clasificación: 'sexo', 'tumor', 'complicacion', 'tipo_complicacion', 'multi'.
            transform (callable, optional): Transformación aplicada al volumen.
            use_monai_io (bool): Si True, se espera que el transform incluya LoadImaged y trabaje con rutas.
        """
        self.root_dir = root_dir
        self.task = task
        self.transform = transform
        self.use_monai_io = use_monai_io
        self.only_labels = only_labels

        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.nii.gz')])

        self.label_maps = {
            'sexo': {},
            'tumor': {},
            'complicacion': {},
            'tipo_complicacion': {},
        }

        self._build_label_maps()

    def _build_label_maps(self):
        categorias = {'sexo': set(), 'tumor': set(), 'complicacion': set(), 'tipo_complicacion': set()}
        for fname in self.files:
            labels = self._parse_filename(fname)
            for cat in categorias:
                categorias[cat].add(labels[cat])

        for cat in categorias:
            self.label_maps[cat] = {val: idx for idx, val in enumerate(sorted(categorias[cat]))}

    def _parse_filename(self, filename):
        if filename.endswith(".nii.gz"):
            fname = filename[:-7]
        elif filename.endswith(".nii"):
            fname = filename[:-4]
        else:
            fname = filename

        # buscamos el sexo (primer no numero)
        for i, c in enumerate(fname):
            if not c.isdigit():
                sexo = c
                tumor_start = i + 1
                break
        else:
            raise ValueError(f"No se encontró sexo en el nombre: {filename}")

        tipo_complicacion = fname[-1]
        complicacion = fname[-2]
        tumor = fname[tumor_start:-2]

        return {
            'sexo': sexo,
            'tumor': tumor,
            'complicacion': complicacion,
            'tipo_complicacion': tipo_complicacion
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.root_dir, fname)

        labels = self._parse_filename(fname)
        mapped_labels = {k: self.label_maps[k][v] for k, v in labels.items()}

        if self.only_labels:
            return mapped_labels if self.task == 'multi' else mapped_labels[self.task]

        if self.use_monai_io:
            sample = {"image": fpath}
            if self.transform:
                sample = self.transform(sample)
            image = sample["image"]
        else:
            img = nib.load(fpath).get_fdata().astype(np.float32)
            img = (img - np.mean(img)) / (np.std(img) + 1e-5)
            img = np.nan_to_num(img)
            if self.transform:
                img = self.transform(img)
            image = torch.tensor(img).unsqueeze(0)  # [1, D, H, W]

        if self.task == 'multi':
            return image, mapped_labels
        elif self.task in mapped_labels:
            return image, mapped_labels[self.task]
        else:
            raise ValueError(f"Tarea no reconocida: {self.task}")


class MaskedLungCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, task='complicacion', transform=None, use_monai_io=False):
        """
        Dataset que carga volúmenes y máscaras, los multiplica y aplica transformaciones.

        Args:
            image_dir (str): Carpeta con volúmenes de TC.
            mask_dir (str): Carpeta con máscaras (mismo nombre base que imagen).
            task (str): Una de {'sexo', 'tumor', 'complicacion', 'tipo_complicacion', 'multi'}.
            transform (callable): Transformación aplicada después de multiplicar imagen * máscara.
            use_monai_io (bool): Si True, transforma rutas directamente (requiere LoadImaged).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.task = task
        self.transform = transform
        self.use_monai_io = use_monai_io

        # construimos mapping base_name 
        self.image_files = self._index_files(image_dir)
        self.mask_files = self._index_files(mask_dir)

        # filtramos solo los que estan en los dos
        common_keys = sorted(set(self.image_files.keys()) & set(self.mask_files.keys()))
        self.pairs = [(self.image_files[k], self.mask_files[k]) for k in common_keys]
        self.filenames = common_keys  # para parsear etiquetas

        # mapeos de etiquetas
        self.label_maps = {k: {} for k in ['sexo', 'tumor', 'complicacion', 'tipo_complicacion']}
        self._build_label_maps()

    def _index_files(self, directory):
        index = {}
        for f in os.listdir(directory):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                base = f.replace(".nii.gz", "").replace(".nii", "")
                index[base] = os.path.join(directory, f)
        return index

    def _parse_filename(self, fname):
        for i, c in enumerate(fname):
            if not c.isdigit():
                sexo = c
                tumor_start = i + 1
                break
        else:
            raise ValueError(f"No se encontro sexo en el nombre: {fname}")
        tipo_complicacion = fname[-1]
        complicacion = fname[-2]
        tumor = fname[tumor_start:-2]
        return {'sexo': sexo, 'tumor': tumor, 'complicacion': complicacion, 'tipo_complicacion': tipo_complicacion}

    def _build_label_maps(self):
        categorias = {k: set() for k in self.label_maps}
        for fname in self.filenames:
            labels = self._parse_filename(fname)
            for cat in categorias:
                categorias[cat].add(labels[cat])
        for cat in categorias:
            self.label_maps[cat] = {val: idx for idx, val in enumerate(sorted(categorias[cat]))}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        fname = self.filenames[idx]
        labels = self._parse_filename(fname)
        mapped_labels = {k: self.label_maps[k][v] for k, v in labels.items()}

        if self.use_monai_io:
            sample = {"image": image_path, "mask": mask_path}
            sample = self.transform(sample) if self.transform else sample
            image = sample["image"] * sample["mask"]
        else:
            image = nib.load(image_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.float32)
            print(mask)
            combined = image * mask
            combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-5)
            combined = np.nan_to_num(combined)
            combined = torch.tensor(combined).unsqueeze(0)  # [1, D, H, W]
            image = self.transform(combined) if self.transform else combined

        if self.task == 'multi':
            return image, mapped_labels
        elif self.task in mapped_labels:
            return image, mapped_labels[self.task]
        else:
            raise ValueError(f"Tarea no reconocida: {self.task}")
        
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class LungCTWithMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, task='complicacion', transform=None,
                 use_monai_io=False, combine_strategy=None, default_value=-2000, return_mask=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.task = task
        self.transform = transform
        self.use_monai_io = use_monai_io
        self.combine_strategy = combine_strategy  # 'stack', 'mask_zero_out', None
        self.default_value = default_value
        self.return_mask = return_mask

        self.files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

        self.label_maps = {
            'sexo': {},
            'tumor': {},
            'complicacion': {},
            'tipo_complicacion': {},
        }

        self._build_label_maps()

    def _build_label_maps(self):
        categorias = {'sexo': set(), 'tumor': set(), 'complicacion': set(), 'tipo_complicacion': set()}
        for fname in self.files:
            labels = self._parse_filename(fname)
            for cat in categorias:
                categorias[cat].add(labels[cat])
        for cat in categorias:
            self.label_maps[cat] = {val: idx for idx, val in enumerate(sorted(categorias[cat]))}

    def _parse_filename(self, filename):
        if filename.endswith(".nii.gz"):
            fname = filename[:-7]
        elif filename.endswith(".nii"):
            fname = filename[:-4]
        else:
            fname = filename

        for i, c in enumerate(fname):
            if not c.isdigit():
                sexo = c
                tumor_start = i + 1
                break
        else:
            raise ValueError(f"No se encontró sexo en el nombre: {filename}")

        tipo_complicacion = fname[-1]
        complicacion = fname[-2]
        tumor = fname[tumor_start:-2]

        return {
            'sexo': sexo,
            'tumor': tumor,
            'complicacion': complicacion,
            'tipo_complicacion': tipo_complicacion
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.image_dir, fname)
        # mask_fname = fname.replace('.nii.gz', '.nii').replace('.nii', '.nii')  # generaliza
        # mask_path = os.path.join(self.mask_dir, mask_fname)
        base = fname.replace('.nii.gz', '').replace('.nii', '')
        mask_path = os.path.join(self.mask_dir, base + '.nii')

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"No se encontró máscara para {fname} → {mask_path}")

        if not os.path.exists(mask_path):
            mask_fname += '.gz'  # fallback por si es .nii.gz
            mask_path = os.path.join(self.mask_dir, mask_fname)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"No se encontró máscara para {fname}")

        labels = self._parse_filename(fname)
        mapped_labels = {k: self.label_maps[k][v] for k, v in labels.items()}

        if self.use_monai_io:
            sample = {"image": fpath, "mask": mask_path}
            if self.transform:
                sample = self.transform(sample)
            if self.return_mask:
                return sample["image"], sample["mask"], mapped_labels[self.task]
            else:
                return sample["image"], mapped_labels[self.task]

        else:
            img = nib.load(fpath).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.float32)

            if self.combine_strategy == "stack":
                combined = np.stack([img, mask], axis=0)  # [2, D, H, W]
            elif self.combine_strategy == "mask_zero_out":
                combined = np.where(mask > 0, img, self.default_value)[None, ...]  # [1, D, H, W]
            else:
                combined = img[None, ...]

            combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-5)
            combined = np.nan_to_num(combined)

            if self.transform:
                combined = self.transform(combined)

            if self.return_mask:
                return  torch.tensor(combined), torch.tensor(mask), mapped_labels[self.task]
            else:
                return torch.tensor(combined), mapped_labels[self.task]
