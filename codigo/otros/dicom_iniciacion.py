#librerías: DICOM, numpy, matplotlib y widgets interactivos
from pydicom import dcmread
import numpy as np
from pydicom.fileset import FileSet
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import defaultdict
import pandas as pd
import seaborn as sns
import os



# Función para visualizar las slices: 
# visualizador 2D con slider para explorar las slices axiales (eje Z)
def plot_slices(volume, patient_name, patient_id, patient_folder):
    # cremoas figura y eje
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Slice inicial
    initial_slice = volume.shape[2] // 2
    slice_view = ax.imshow(volume[:, :, initial_slice], cmap="gray")
    
    # título con el nombre, ID y carpeta del paciente
    ax.set_title(f"Paciente: {patient_name} (ID: {patient_id})\nCarpeta: {patient_folder} (Axial View)")

    # slider para explorar las slices
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Slice", 0, volume.shape[2] - 1, valinit=initial_slice, valstep=1)

    # función para actualizar la slice mostrada
    def update(val):
        slice_idx = int(slider.val)
        slice_view.set_data(volume[:, :, slice_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show() #mostramos


# Función para dibujar las slices en cualquier eje
def plot_3d_slices(volume):
    # Initialize the figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Default axis and slice index
    current_axis = 2  # 0 for axial, 1 for coronal, 2 for sagittal
    initial_slice = volume.shape[current_axis] // 2

    # Display the initial slice along the default axis (axial view)
    slice_view = ax.imshow(volume[:, :, initial_slice], cmap="gray")
    ax.set_title("Axial View (Z-Axis)")

    # create slider to navigate slices
    ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Slice", 0, volume.shape[current_axis] - 1, valinit=initial_slice, valstep=1)

    # button axes and buttons for axis selection
    ax_button_axial = plt.axes([0.25, 0.1, 0.1, 0.04])
    ax_button_coronal = plt.axes([0.45, 0.1, 0.1, 0.04])
    ax_button_sagittal = plt.axes([0.65, 0.1, 0.1, 0.04])

    button_axial = Button(ax_button_axial, "Axial")
    button_coronal = Button(ax_button_coronal, "Coronal")
    button_sagittal = Button(ax_button_sagittal, "Sagital")

    # update function to display the correct slice when slider moves
    def update_slice(val):
        slice_idx = int(slider.val)
        if current_axis == 0:
            slice_view.set_data(volume[slice_idx, :, :])
            ax.set_title("Axial View (Z-Axis)")
        elif current_axis == 1:
            slice_view.set_data(volume[:, slice_idx, :])
            ax.set_title("Coronal View (Y-Axis)")
        elif current_axis == 2:
            slice_view.set_data(volume[:, :, slice_idx])
            ax.set_title("Sagittal View (X-Axis)")
        fig.canvas.draw_idle()

    #function to reset slider when switching between views
    def reset_slider(axis):
        nonlocal current_axis
        current_axis = axis
        slider.valmin = 0
        slider.valmax = volume.shape[current_axis] - 1
        slider.set_val(volume.shape[current_axis] // 2)  # Reset slider to middle slice
        slider.ax.set_xlim(slider.valmin, slider.valmax)  # Update slider range
        update_slice(slider.val)

    #functions to switch between axial, coronal, and sagittal views
    def set_axial(event):
        reset_slider(0)

    def set_coronal(event):
        reset_slider(1)

    def set_sagittal(event):
        reset_slider(2)

    # Connect buttons to their functions
    button_axial.on_clicked(set_axial)
    button_coronal.on_clicked(set_coronal)
    button_sagittal.on_clicked(set_sagittal)

    # Connect slider update to the update_slice function
    slider.on_changed(update_slice)

    # Show the plot
    plt.show()


#lectura del csv
train = pd.read_csv("datos_limpios.csv", na_values="NaN", sep = ",") # na_values para identificar bien los valores perdidos
train.info()

#atributos
train.columns

def main():
    #path = "./datosprueba/casosradiomica/1MASh/DICOMDIR"
    # path = "./casosradiomica/1MASh/DICOMDIR" #miramos el primer ejemplo

    # # leemos el dicomdir
    # dsdir = dcmread(path)

    # fs = FileSet(dsdir)

    # print(fs)

    # # accedemos a los datos y dibujando
    # for i,instance in enumerate(fs): #recorremos todos los archivos de la carpeta
    #     print(instance.PatientName, instance.PatientID, instance.SOPInstanceUID)
    #     # Load the instance
    #     ds = instance.load()

    #     #imprimimos los metadatos básicos: nombre paciente, ID...
    #     print(ds.PhotometricInterpretation)
    #     print(ds.pixel_array.shape)

    #     if i == 400: #mostramos una slice especificamente como prueba
    #         # Show the image
    #         plt.imshow(ds.pixel_array, cmap='gray')
    #         plt.show()

    # # agrupamos por pacientes
    # patient_slices = defaultdict(list)
    # for i, instance in enumerate(fs):
    #     ds = instance.load()
    #     patient_slices[instance.PatientID].append(ds)

    # for patient_id, slices in patient_slices.items():
    #     print(f"Patient ID {patient_id} has {len(slices)} slices")

    # # Sort the slices
    # for patient_id, slices in patient_slices.items():
    #     patient_slices[patient_id] = sorted(slices, key=lambda s: s.SliceLocation, reverse=True)

    # # Create a 3D volume
    # for patient_id, slices in patient_slices.items():
    #     volume = np.stack([s.pixel_array for s in slices], axis=2)
    #     print(volume.shape)

    #     # Show th swrast: /usr/lib/dri/swre volume with a slider to navigate through slices
    #     plot_slices(volume)

    #     # Show the volume with a slider to navigate through slices in any axis
    #     plot_3d_slices(volume)

    #     break


    root_path = "./casosradiomica/" 
    patient_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    for patient_dir in patient_dirs:
        dicomdir_path = os.path.join(patient_dir, "DICOMDIR")
        if not os.path.exists(dicomdir_path):
            print(f"No se encontró DICOMDIR en el directorio {patient_dir}. Se omite.")
            continue

        try:
            # leemos el archivo dicom
            dsdir = dcmread(dicomdir_path)

            # FileSet para el paciente
            fs = FileSet(dsdir)

            # agrupamos slices por paciente
            patient_slices = []
            patient_name = None
            patient_id = None
            for instance in fs:
                ds = instance.load()
                patient_slices.append(ds)

                # obtenemos el nombre y el ID del paciente (si no se ha obtenido previamente)
                if patient_name is None or patient_id is None:
                    patient_name = ds.PatientName if hasattr(ds, 'PatientName') else 'Desconocido'
                    patient_id = ds.PatientID if hasattr(ds, 'PatientID') else 'Desconocido'

            # ordenamos slices por posición
            patient_slices = sorted(patient_slices, key=lambda s: getattr(s, "SliceLocation", 0))

            # creamos volumen 3D
            volume = np.stack([s.pixel_array for s in patient_slices], axis=2)

            # nombre de la carpeta como ID de la carpeta
            patient_folder = os.path.basename(patient_dir)

            # mostramos el volumen con slider
            print(f"Procesando Paciente: {patient_name} (ID: {patient_id}), Carpeta: {patient_folder}, Volume shape: {volume.shape}")
            plot_slices(volume, patient_name, patient_id, patient_folder)

        except Exception as e:
            print(f"Error al procesar el paciente en el directorio {patient_dir}: {e}")

if __name__ == "__main__":
    main()
