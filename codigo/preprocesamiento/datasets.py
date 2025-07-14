import torch
import os
import json
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annos = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        anno_path = os.path.join(self.root, "annotations", self.annos[idx])

        img = Image.open(img_path).convert("RGB")
        with open(anno_path) as f:
            anno = json.load(f)

        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["labels"], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


class VOCDatasetV1(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, classes=None):
        """
        Args:
            root (str): Root directory of the dataset.
                        Expected structure:
                        root/
                        ├── images/
                        └── annotations/
            transforms (callable, optional): A function/transform to apply to the images.
            classes (list, optional): List of class names in the dataset.
                                      The first class must be "__background__".
                                      Example: ["__background__", "fronton", "fronton-curvo", ...]
        """
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annos = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def parse_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.class_to_idx:
                continue  # Skip unknown labels
            labels.append(self.class_to_idx[label])

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        anno_path = os.path.join(self.root, "annotations", self.annos[idx])
        boxes, labels = self.parse_xml(anno_path)

        # Build the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


KEY_ELEMENTS_MONUMAI = [
    "__background__",
    'fronton-partido',
    'vano-adintelado',
    'fronton',
    'arco-medio-punto',
    'columna-salomonica',
    'ojo-de-buey',
    'fronton-curvo',
    'serliana',
    'arco-apuntado',
    'pinaculo-gotico',
    'arco-conopial',
    'arco-trilobulado',
    'arco-herradura',
    'dintel-adovelado',
    'arco-lobulado'
]

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, classes=KEY_ELEMENTS_MONUMAI):
        """
        Args:
            root (str): Root directory of the dataset.
                        Expected structure:
                        root/
                        ├── images/
                        └── annotations/
            transforms (callable, optional): A function/transform to apply to the images.
            classes (list, optional): List of class names in the dataset.
                                      The first class must be "__background__".
                                      Example: ["__background__", "fronton", "fronton-curvo", ...]
        """
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Ensure sorted order for consistency
        self.imgs = sorted(f for f in os.listdir(os.path.join(root, "images")) if f.endswith(".jpg"))
        self.annos = sorted(f for f in os.listdir(os.path.join(root, "annotations")) if f.endswith(".xml"))

        # Check that images and annotations match
        assert len(self.imgs) == len(self.annos), "Mismatch between images and annotations."
        for img, anno in zip(self.imgs, self.annos):
            assert os.path.splitext(img)[0] == os.path.splitext(anno)[0], "Image and annotation file names do not match."

    def parse_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.class_to_idx:
                continue  # Skip unknown labels
            labels.append(self.class_to_idx[label])

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        # Return empty tensors if no valid objects
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), labels

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # Load annotations
        anno_path = os.path.join(self.root, "annotations", self.annos[idx])
        boxes, labels = self.parse_xml(anno_path)

        # Build the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,  # torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # print ("Prior boxes: ", boxes)
        # print ("Prior labels: ", labels)
        # if self.transforms:
        #     img = self.transforms(img)

        # Apply albumentations transform if provided
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']  # already converted to tensor by ToTensorV2
            boxes = transformed['bboxes']
            labels = transformed['labels']
            
            # Update target: convert boxes and labels to torch tensors
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            
            # Optionally recompute area if needed (or let your collate function handle it)
            if len(boxes) > 0:
                boxes_np = np.array(boxes)
                target["area"] = torch.as_tensor((boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1]), dtype=torch.float32)
            else:
                target["area"] = torch.tensor([])
            
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # print ("Post boxes: ", target["boxes"])
        # print ("Post labels: ", target["labels"])

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class StyleClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
                        Expected structure:
                        root/
                        ├── images/
                        └── annotations/
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annos = list(sorted(os.listdir(os.path.join(root, "annotations"))))

        # Extract the list of all unique labels from XML annotations
        self.labels = self._get_labels()

        # Create a mapping from label names to integer indices
        self.class_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def _get_labels(self):
        """Extracts unique labels from the XML annotations."""
        labels = set()
        for anno_file in self.annos:
            xml_path = os.path.join(self.root, "annotations", anno_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the <style> tag content as the label
            style_tag = root.find("style")
            if style_tag is not None:
                labels.add(style_tag.text)

        return sorted(labels)

    def __getitem__(self, idx):
        # Load image
        img_file = self.imgs[idx]
        img_path = os.path.join(self.root, "images", img_file)
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        anno_file = self.annos[idx]
        anno_path = os.path.join(self.root, "annotations", anno_file)
        tree = ET.parse(anno_path)
        root = tree.getroot()

        # Extract the <style> tag content as the label
        style_tag = root.find("style")
        if style_tag is not None:
            label = style_tag.text
        else:
            raise ValueError(f"Style tag missing in annotation {anno_file}")

        # Convert label to index
        label_idx = self.class_to_idx[label]

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return img, label_idx

    def __len__(self):
        return len(self.imgs)
    



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

        # Buscar sexo (primer no dígito)
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

        # Construir mapping base_name → ruta imagen y máscara
        self.image_files = self._index_files(image_dir)
        self.mask_files = self._index_files(mask_dir)

        # Filtrar solo los que están en ambos
        common_keys = sorted(set(self.image_files.keys()) & set(self.mask_files.keys()))
        self.pairs = [(self.image_files[k], self.mask_files[k]) for k in common_keys]
        self.filenames = common_keys  # para parsear etiquetas

        # Mapeos de etiquetas
        self.label_maps = {k: {} for k in ['sexo', 'tumor', 'complicacion', 'tipo_complicacion']}
        self._build_label_maps()

    def _index_files(self, directory):
        """Mapea nombre base (sin extensión) → ruta absoluta"""
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
            raise ValueError(f"No se encontró sexo en el nombre: {fname}")
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
