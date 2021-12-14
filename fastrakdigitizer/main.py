# pip install pyserial
# sudo usermod -a -G tty $USER
# sudo usermod -a -G dialout $USER
# Need to re-log for the groups to be updated

import serial
import sys
from PySide2 import QtWidgets, QtCore, Qt3DRender, QtGui
from PySide2.Qt3DCore import Qt3DCore
from PySide2.Qt3DExtras import Qt3DExtras
from PySide2.Qt3DRender import Qt3DRender

import open3d
import configparser

import os
import beepy
import mne
from pathlib import Path
import numpy as np
import pandas as pd

from mne.io.constants import FIFF
from mne.transforms import _get_trans, combine_transforms, Transform, apply_trans, transform_surface_to
from mne.viz._3d import _fiducial_coords
from mne.utils import get_subjects_dir
from mne.bem import read_bem_surfaces
from mne.surface import _project_onto_surface
import trimesh


config_path = str(Path(__file__).parent.absolute() / 'config.ini')


def recover_homogenous_affine_trans(points1, points2):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = points1[1:] - points1[0]
    Q_prime = points2[1:] - points2[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = points2[0] - np.dot(points1[0], R)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1))).T


def apply_transform(points, trans_mat):
    # If trans_mat is obtained with recover_homogenous_affine_trans(points1, points2),
    # then apply_transform(points1, trans_mat) == points2
    points = np.array(points)
    a = np.hstack([points, np.ones((points.shape[0], 1))])
    ap = np.dot(trans_mat, a.T).T[:, :3]
    return ap


def get_movement_correct_dig(ref_triad, dig_triad, dig):
    """
     Using three reference point to correct for movements
     during acquisition.
     ref_triad: a 3x3 matrix of three points used as a reference.
     dig_triad: a 3x3 matrix of the position for these three points, possibly moved in spaced comparted to the ref_triad
     dig: a 1x3 point digitized as the same time as the dig_triad
    """

    trans_mat = recover_homogenous_affine_trans(dig_triad, ref_triad)
    return apply_transform(dig, trans_mat)


def correct_for_movements(df_in):
    df = df_in.copy()
    ref_triad = df.iloc[0, 3:].values.reshape((3, 3))
    for i in range(1, df.shape[0]):
        dig_triad = df.iloc[i, 3:].values.reshape((3, 3))
        dig = [df.iloc[i, :3].values]
        df.iloc[i, :3] = get_movement_correct_dig(ref_triad, dig_triad, dig)[0]
        df.iloc[i, 3:] = ref_triad.reshape([9])
    return df


class DataFrameModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.Property(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.Slot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return None

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return 3 #self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() and 0 <= index.column() < self.columnCount()):
            return None
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.loc[row, col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return None

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles


def sphere_mesh(pos, radius, color=None):
    mesh = trimesh.creation.icosphere(radius=radius, color=color)
    mesh.vertices += pos
    return mesh


def get_aligned_artifacts(info=None, trans=None, subject=None, subjects_dir=None,
                          coord_frame='mri', head_surf=None):
    head_mri_t, _ = _get_trans(trans, 'head', 'mri')
    dev_head_t, _ = _get_trans(info['dev_head_t'], 'meg', 'head')
    head_trans = head_mri_t
    mri_trans = Transform('mri', 'mri')

    mri_fiducials = mne.coreg.get_mni_fiducials(subject, subjects_dir)
    fid_loc = _fiducial_coords(mri_fiducials, FIFF.FIFFV_COORD_MRI)
    fid_loc = apply_trans(mri_trans, fid_loc)
    fid_loc = pd.DataFrame(fid_loc, index=[fid["ident"]._name.split("_")[-1] for fid in mri_fiducials],
                           columns=["x", "y", "z"])

    if head_surf is None:
        subject_dir = Path(get_subjects_dir(subjects_dir, raise_error=True)) / subject
        fname = subject_dir / 'bem' / 'sample-head.fif'
        head_surf = read_bem_surfaces(fname)[0]
        head_surf = transform_surface_to(head_surf, coord_frame, [mri_trans, head_trans], copy=True)

    eeg_picks = mne.pick_types(info, meg=False, eeg=True, ref_meg=False)
    eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_loc = apply_trans(head_trans, eeg_loc)
    eegp_loc = _project_onto_surface(eeg_loc, head_surf, project_rrs=True, return_nn=True)[2]
    eegp_loc = pd.DataFrame(eegp_loc, index=[ch["ch_name"] for ch in info['chs']], columns=["x", "y", "z"])

    return eegp_loc, fid_loc, head_surf


class ComThread(QtCore.QThread):
    command_received = QtCore.Signal(list)
    text_read = QtCore.Signal(str)
    connection_failed = QtCore.Signal(str)

    def __init__(self, receive_wgt, input_wgt, port='/dev/ttyUSB0'):
        QtCore.QThread.__init__(self)
        self.receive_wgt = receive_wgt
        self.input_wgt = input_wgt
        self.port = port
        self.continue_running = True

    def run(self):
        def split_command(command):
            return [command[0:3].strip(),
                    command[3:10].strip(),
                    command[10:17].strip(),
                    command[17:24].strip()]

        serial_kwrds = dict(port=self.port,
                            baudrate=115200,
                            timeout=1,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS)

        record_ids = ["01", "02", "03", "04"]
        buffer = []

        if not Path(self.port).exists():
            self.connection_failed.emit(f"The port ({self.port}) you have selected does not exist.")
            return

        with serial.Serial(**serial_kwrds) as ser:
            while ser.isOpen() and self.continue_running:

                command = self.input_wgt.get_command()
                if command is not None:
                    ser.write(str.encode(f'{command}\n'))
                    ser.flush()
                    self.msleep(100)
                    # let's wait one second before reading output (let's give device time to answer)

                # Read a blocks of 4 lines, one for each sensor
                out = ser.readline().decode()
                if len(out):
                    self.text_read.emit(out)
                    tokens = split_command(out)
                    if tokens[0][:2] == record_ids[len(buffer)]:
                        buffer.append(tokens)
                    if len(buffer) == 4:
                        self.command_received.emit(buffer)
                        # Wait 0.1 second and flush to make sure double button press are not
                        # entered as two digitization
                        ser.flush()
                        self.msleep(100)
                        buffer = []


class CmdInputWidget(QtWidgets.QWidget):

    disconnect_signal = QtCore.Signal()
    connect_signal = QtCore.Signal()

    def __init__(self, parent):
        super(CmdInputWidget, self).__init__(parent)
        self.config = parent.config
        self.init_ui()
        self.command_lst = []

    def init_ui(self):

        vbox = QtWidgets.QVBoxLayout()

        self.connect_btn = QtWidgets.QPushButton("Connect", self)
        self.connect_btn.clicked.connect(self.connect_rs233)

        self.disconnect_btn = QtWidgets.QPushButton("Disconnect", self)
        self.disconnect_btn.clicked.connect(self.disconnect_rs233)
        self.disconnect_btn.setEnabled(False)

        if "rs232_port" in self.config["DEFAULT"]:
            port = self.config["DEFAULT"]["rs232_port"]
        else:
            port = '/dev/ttyUSB0'

        self.rs232_port_wgt = QtWidgets.QLineEdit(port, self)
        self.rs232_port_wgt.textEdited.connect(self.port_text_changed)

        rs232_wgt = QtWidgets.QHBoxLayout()
        rs232_wgt.addWidget(self.connect_btn)
        rs232_wgt.addWidget(self.disconnect_btn)
        rs232_wgt.addWidget(QtWidgets.QLabel("Serial port:", self))
        rs232_wgt.addWidget(self.rs232_port_wgt)

        self.cmd_input = QtWidgets.QLineEdit(self)
        send_btn = QtWidgets.QPushButton("Send", self)
        send_btn.clicked.connect(self.send_button_clicked)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("RS-232 command:"))
        hbox.addWidget(self.cmd_input)
        hbox.addWidget(send_btn)

        vbox.addLayout(rs232_wgt)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def port_text_changed(self, port):
        self.config["DEFAULT"]["rs232_port"] = port
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)

    def connect_rs233(self):
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.connect_signal.emit()

    def disconnect_rs233(self):
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_signal.emit()

    def send_button_clicked(self):
        command = self.cmd_input.text()
        if len(command):
            self.command_lst.append(command)
            self.cmd_input.setText("")

    def get_command(self):
        if len(self.command_lst):
            return self.command_lst.pop(0)
        return None


class CmdReceivedWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CmdReceivedWidget, self).__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.received_cmds = QtWidgets.QTextEdit(self)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.received_cmds)
        self.setLayout(vbox)

    def command_received(self, text):
         self.received_cmds.insertPlainText(text)
         pos = self.received_cmds.verticalScrollBar().maximum()
         self.received_cmds.verticalScrollBar().setValue(pos)


class MeshDisplayWidget(QtWidgets.QSplitter):
    def __init__(self, parent, fs_subject="sample"):
        super(MeshDisplayWidget, self).__init__(parent)

        self.config = parent.config
        self.fs_subject = fs_subject

        self.view = Qt3DExtras.Qt3DWindow()
        self.view.defaultFrameGraph().setClearColor(QtGui.QColor("#4d4d4f"))
        self.container = QtWidgets.QWidget.createWindowContainer(self.view, self)
        screen_size = self.view.screen().size()
        self.container.setMinimumSize(QtCore.QSize(600, 100))
        self.container.setMaximumSize(screen_size)

        #h_layout = QtWidgets.QHBoxLayout(self)
        #h_layout.addWidget(self.container)
        self.addWidget(self.container)

        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setObjectName("tableView")

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(self.tableView)

        self.overlay_btn = QtWidgets.QPushButton("Update and overlay digitized points")
        self.overlay_btn.clicked.connect(self.update_acquisition_overlay)
        v_layout.addWidget(self.overlay_btn)

        subject_name_layout = QtWidgets.QHBoxLayout()
        subject_name_layout.addWidget(QtWidgets.QLabel("Subject name:"))
        self.subject_name_wgt = QtWidgets.QComboBox()
        subject_name_layout.addWidget(self.subject_name_wgt)
        self.new_subject_wgt = QtWidgets.QPushButton("New subject")
        self.new_subject_wgt.clicked.connect(self.new_subject)
        subject_name_layout.addWidget(self.new_subject_wgt)

        self.mute_sound_wgt = QtWidgets.QCheckBox("Mute sound")
        subject_name_layout.addWidget(self.mute_sound_wgt)

        v_layout.addLayout(subject_name_layout)

        if "subject_dir" not in self.config["DEFAULT"]:
            self.config["DEFAULT"]["subject_dir"] = str(Path(__file__).parent.absolute() / "subjects_data")

        subject_dir_layout = QtWidgets.QHBoxLayout()
        subject_dir_layout.addWidget(QtWidgets.QLabel("Subject dir:"))
        self.subject_dir_wgt = QtWidgets.QLineEdit(self.config["DEFAULT"]["subject_dir"])
        self.subject_dir_wgt.setReadOnly(False)
        subject_dir_layout.addWidget(self.subject_dir_wgt)
        select_subject_dir_btn = QtWidgets.QPushButton("...")
        subject_dir_layout.addWidget(select_subject_dir_btn)
        select_subject_dir_btn.clicked.connect(self.open_subject_dir_dlg)

        v_layout.addLayout(subject_dir_layout)
        self.subject_dir_wgt.textChanged.connect(self.set_subject_dir)

        if not Path(self.config["DEFAULT"]["subject_dir"]).exists():
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Information)
            msgBox.setText(f"The chosen subject directory (i.e. {self.config['DEFAULT']['subject_dir']}) does not "
                           "exist. Do you want to create it or to select a new directory?")
            msgBox.setWindowTitle("Non-existent subject directory")

            button_create = msgBox.addButton("Create it!", QtWidgets.QMessageBox.YesRole);
            msgBox.addButton("Select a different directory", QtWidgets.QMessageBox.NoRole)
            msgBox.exec_()
            if msgBox.clickedButton() == button_create:
                Path(self.config["DEFAULT"]["subject_dir"]).mkdir(exist_ok=True, parents=True)
            else:
                self.open_subject_dir_dlg()

        #h_layout.addLayout(v_layout)
        v_layout_widget = QtWidgets.QWidget(self)
        v_layout_widget.setLayout(v_layout)
        self.addWidget(v_layout_widget)

        self.data = QtCore.QUrl.fromLocalFile(str(Path(__file__).parent.absolute() / f"{self.fs_subject}_head.obj"))

        self.root_entity = Qt3DCore.QEntity()

        self.material = Qt3DExtras.QPhongMaterial()
        self.material.setDiffuse(QtGui.QColor(254, 254, 254))

        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(400.0, 16.0/9.0, 100, 1200.0)
        self.camera.setPosition(QtGui.QVector3D(0, 500, 0))
        self.camera.setUpVector(QtGui.QVector3D(0, 0, 1))

        self.lightEntities = {}
        self.lights = {}
        self.lightTransforms = {}
        for trans_x in [200, -200]:
            for trans_y in [200, -200]:
                self.lightEntities[(trans_x, trans_y)] = Qt3DCore.QEntity(self.root_entity)
                self.lights[(trans_x, trans_y)] = Qt3DRender.QPointLight(self.lightEntities[(trans_x, trans_y)])
                self.lights[(trans_x, trans_y)].setColor("white")
                self.lights[(trans_x, trans_y)].setIntensity(0.4)
                self.lightEntities[(trans_x, trans_y)].addComponent(self.lights[(trans_x, trans_y)])
        
                self.lightTransforms[(trans_x, trans_y)] = Qt3DCore.QTransform(self.lightEntities[(trans_x, trans_y)])
                self.lightTransforms[(trans_x, trans_y)].setTranslation(QtGui.QVector3D(trans_x, trans_y, 100.0))
                self.lightEntities[(trans_x, trans_y)].addComponent(self.lightTransforms[(trans_x, trans_y)])

        # Get co-registered head, fiducials and electrodes
        data_path = Path(mne.datasets.sample.data_path())
        fs_subjects_dir = data_path / 'subjects'
        trans_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'
        trans = mne.read_trans(trans_fname)

        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        info = mne.create_info(montage.ch_names, 100, ch_types='eeg')
        raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 100)), info)
        raw.set_montage(montage)

        eeg, fid, surf = get_aligned_artifacts(raw.info, subject=self.fs_subject, trans=trans,
                                               subjects_dir=fs_subjects_dir, coord_frame='mri')
        self.head_surf = surf

        path_out = Path(str(Path(__file__).parent.absolute() / f"{self.fs_subject}_head.obj"))
        if True: #not path_out.exists():
            mesh = trimesh.Trimesh(surf["rr"] * 1000, surf["tris"])
            open3d_mesh = open3d.geometry.TriangleMesh(vertices=open3d.utility.Vector3dVector(mesh.vertices),
                                                       triangles=open3d.utility.Vector3iVector(mesh.faces))
            mesh = open3d_mesh.simplify_quadric_decimation(int(20000))
            mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

            with path_out.open('w') as file_obj:
                file_obj.write(trimesh.exchange.obj.export_obj(mesh))

        # Head
        self.data = QtCore.QUrl.fromLocalFile(str(path_out))

        self.head_entity = Qt3DCore.QEntity(self.root_entity)
        self.head_mesh = Qt3DRender.QMesh()
        self.head_mesh.setMeshName(str(path_out))
        self.head_mesh.setSource(self.data)

        self.head_entity.addComponent(self.head_mesh)
        self.head_entity.addComponent(self.material)

        self.camController = Qt3DExtras.QOrbitCameraController(self.head_mesh)
        self.camController.setCamera(self.camera)
        self.camController.setLinearSpeed(self.camController.linearSpeed()*100)
        self.mesh_center_of_mass = np.median(surf['rr'], 0)*1000
        self.camera.setViewCenter(QtGui.QVector3D(*(self.mesh_center_of_mass)))

        self.camera.viewCenterChanged.connect(self.cam_view_center_changed)

        ## EEG electrodes and Fiducials
        self.electrode_material = Qt3DExtras.QPhongMaterial(diffuse=QtGui.QColor("#665423"))
        self.fid_material = Qt3DExtras.QPhongMaterial(diffuse=QtGui.QColor("#0000FF"))
        self.selected_material = Qt3DExtras.QPhongMaterial(diffuse=QtGui.QColor("#FF0000"))
        self.edited_material = Qt3DExtras.QPhongMaterial(diffuse=QtGui.QColor("#FFFF00"))
        self.acq_material = Qt3DExtras.QPhongMaterial(diffuse=QtGui.QColor("#00FF00"))
        self.selected_item = None

        self.coordinate_meshes = {}
        self.coordinate_entities = {}
        self.coordinate_transforms = {}
        self.coordinate_materials = {}

        for kind, var in zip(["fid", "elec"], [fid, eeg]):
            for label, pos in var.iterrows():
                pos = pos.values*1000
                self.coordinate_meshes[label] = Qt3DExtras.QSphereMesh(rings=20, slices=20, radius=3)
                self.coordinate_entities[label] = Qt3DCore.QEntity(self.root_entity)
                self.coordinate_transforms[label] = Qt3DCore.QTransform(self.coordinate_meshes[label])
                self.coordinate_transforms[label].setTranslation(QtGui.QVector3D(*pos))

                self.coordinate_entities[label].addComponent(self.coordinate_meshes[label])
                self.coordinate_entities[label].addComponent(self.coordinate_transforms[label])

        self.acq_coordinate_meshes = {}
        self.acq_coordinate_entities = {}
        self.acq_coordinate_transforms = {}

        self.df_montage = pd.concat([fid, eeg])
        self.df_montage.columns = ["x", "y", "z"]
        self.df_acq = pd.concat([self.df_montage.copy().rename(columns={"x": f"x{i+1}", "y": f"y{i+1}", "z": f"z{i+1}"})
                                 for i in range(4)], axis=1)
        model = DataFrameModel(self.df_acq)
        self.tableView.setModel(model)
        self.df_acq.loc[:, :] = np.nan

        self.set_table_view_selection(0)
        self.view.setRootEntity(self.root_entity)

        self.subject_name_wgt.currentTextChanged.connect(self.load_subject)
        self.update_subject_lst()

    def update_channel_material_all(self):
        for label in self.coordinate_entities:
            self.update_channel_material(label)

    def update_channel_material(self, label):
        if label in self.coordinate_materials:
            self.coordinate_entities[label].removeComponent(self.coordinate_materials[label])

        if label == self.selected_item:
            self.coordinate_materials[label] = self.selected_material
        elif not np.any(np.isnan(self.df_acq.loc[label].values)):
            self.coordinate_materials[label] = self.edited_material
        elif label in ["NASION", "LPA", "RPA"]:
            self.coordinate_materials[label] = self.fid_material
        else:
            self.coordinate_materials[label] = self.electrode_material

        self.coordinate_entities[label].addComponent(self.coordinate_materials[label])

    def set_table_view_selection(self, index=0):
        self.selection = self.tableView.selectionModel()
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.selection.selectionChanged.connect(self.handleSelectionChanged)
        self.tableView.selectRow(index)

    def update_subject_lst(self, index=None, subject_name=None):
        self.check_subject_directory()
        existing_subjects = sorted(Path(self.subject_dir_wgt.text()).glob("*_df_acq.csv"))
        existing_subjects = [str(path.name)[:-11] for path in existing_subjects]
        if len(existing_subjects):
            self.subject_name_wgt.clear()
            self.subject_name_wgt.addItems(existing_subjects)
        else:
            self.new_subject()
            return

        if subject_name is not None:
            self.subject_name_wgt.setCurrentText(subject_name)
        elif index is not None:
            self.subject_name_wgt.setCurrentIndex(index)
        else:
            self.subject_name_wgt.setCurrentIndex(self.subject_name_wgt.count()-1)

    def new_subject(self):
        existing_subjects = sorted(Path(self.config["DEFAULT"]["subject_dir"])
                                   .glob("subject[0-9][0-9][0-9][0-9][0-9]_df_acq.csv"))
        existing_subjects = [str(path.name)[:-11] for path in existing_subjects]
        if len(existing_subjects) == 0:
            subject_name = "subject00001"
        else:
            subject_name = f"subject{int(existing_subjects[-1][8:13])+1:05}"

        subject_name, ok = QtWidgets.QInputDialog.getText(self, 'text', 'Enter some text',
                                                          QtWidgets.QLineEdit.Normal, subject_name)
        if ok:
            if Path(f"{self.config['DEFAULT']['subject_dir']}/{subject_name}_df_acq.csv").exists():
                msgBox = QtWidgets.QMessageBox()
                msgBox.setIcon(QtWidgets.QMessageBox.Information)
                msgBox.setText(f"This subject ({subject_name}) already exists. Do you want to load its data "
                               f"or do you want to choose a different name?")
                msgBox.setWindowTitle("Already existing subject")

                msgBox.addButton("Load this subject", QtWidgets.QMessageBox.YesRole);
                button_choose = msgBox.addButton("Choose a different name", QtWidgets.QMessageBox.NoRole)
                msgBox.exec_()
                if msgBox.clickedButton() == button_choose:
                    self.new_subject()
                    return

                self.update_subject_lst(subject_name=subject_name)
                return

            self.df_acq.loc[:, :] = np.nan
            self.df_acq.to_csv(f"{self.config['DEFAULT']['subject_dir']}/{subject_name}_df_acq.csv")
            self.update_subject_lst(subject_name=subject_name)

    def load_subject(self, subject_name):
        if subject_name == "":
            return
        self.df_acq = pd.read_csv(f"{self.config['DEFAULT']['subject_dir']}/{subject_name}_df_acq.csv",
                                  index_col=0, header=0)
        model = DataFrameModel(self.df_acq)
        self.tableView.setModel(model)
        self.set_table_view_selection(0)
        self.update_channel_material_all()
        self.clear_acq_spheres()

    def check_subject_directory(self):
        if not Path(self.subject_dir_wgt.text()).exists():
            QtWidgets.QMessageBox.warning(self, "Warning", "Invalid subject directory. This directory does not exist.")
            self.open_subject_dir_dlg()
            return

        if not os.access(self.subject_dir_wgt.text(), os.W_OK):
            QtWidgets.QMessageBox.warning(self, "Warning", "Invalid subject directory: You don't have "
                                                           "write access to this directory.")
            self.open_subject_dir_dlg()
            return

    def open_subject_dir_dlg(self):
        flags = QtWidgets.QFileDialog.DontResolveSymlinks | QtWidgets.QFileDialog.ShowDirsOnly
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Directory", os.getcwd(), flags)
        self.subject_dir_wgt.setText(directory)
        self.check_subject_directory()
        self.update_subject_lst()

    @property
    def subject(self):
        return self.subject_name_wgt.currentText()

    @QtCore.Slot(str)
    def set_subject_dir(self):
        self.config["DEFAULT"]["subject_dir"] = self.subject_dir_wgt.text()
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)

    def cam_view_center_changed(self):
        self.camera.setViewCenter(QtGui.QVector3D(*self.mesh_center_of_mass))
        #self.camera.translateWorld(QtGui.QVector3D(*np.array([0, 0, 100])), Qt3DRender.QCamera.TranslateViewCenter)

    def handleSelectionChanged(self, selected, deselected):
        index = self.selection.selectedRows()[0]
        self.change_select_coordinate(self.df_montage.iloc[index.row()], self.df_montage.index[index.row()])

    def change_select_coordinate(self, pos, label):
        cam_pos = pos/np.linalg.norm(pos)*500 + pos
        self.camera.setPosition(QtGui.QVector3D(*cam_pos))
        self.camera.setViewCenter(QtGui.QVector3D(*pos))

        past_selected = None
        if self.selected_item is not None:
            past_selected = self.selected_item
        self.selected_item = label

        self.update_channel_material(self.selected_item)
        if past_selected is not None:
            self.update_channel_material(past_selected)

    def command_received(self, commands):
        for command, i in zip(commands, range(4)):
            assert (command[0][:2] == f"0{i+1}")
            self.df_acq.loc[self.selected_item,
                            [f"x{i+1}", f"y{i+1}", f"z{i+1}"]] = np.array(command[1:4], dtype=float)/100  # cm to m

        if not self.mute_sound_wgt.isChecked():
            beepy.beep(1)
        self.df_acq.to_csv(f"{self.config['DEFAULT']['subject_dir']}/{self.subject}_df_acq.csv")
        self.tableView.selectRow(self.selection.selectedRows()[0].row()+1)

    def update_acquisition_overlay(self):
        if not np.any(np.isnan(self.df_acq.loc[["NASION", "LPA", "RPA"]].values)):
            self.save_montage_and_trans()
            self.add_acq_coordinates()

    def clear_acq_spheres(self):
        for label in self.acq_coordinate_meshes:
            self.root_entity.removeComponent(self.acq_coordinate_meshes[label])
        self.acq_coordinate_meshes = {}
        self.acq_coordinate_entities = {}
        self.acq_coordinate_transforms = {}

    def add_acq_sphere(self, df, label):
        pos = df.loc[label, ["x", "y", "z"]].values*1000  # m to mm
        if label not in self.acq_coordinate_entities:
            self.acq_coordinate_meshes[label] = Qt3DExtras.QSphereMesh(rings=20, slices=20, radius=3)
            self.acq_coordinate_entities[label] = Qt3DCore.QEntity(self.root_entity)
            self.acq_coordinate_transforms[label] = Qt3DCore.QTransform(self.acq_coordinate_meshes[label])

            self.acq_coordinate_entities[label].addComponent(self.acq_coordinate_meshes[label])
            self.acq_coordinate_entities[label].addComponent(self.acq_coordinate_transforms[label])
            self.acq_coordinate_entities[label].addComponent(self.acq_material)

        self.acq_coordinate_transforms[label].setTranslation(QtGui.QVector3D(*pos))

    def add_acq_coordinates(self):
        data_path = Path(mne.datasets.sample.data_path())
        fs_subjects_dir = data_path / 'subjects'

        montage = self.get_acq_montage()
        print("montage", montage)
        if montage is None:
            return None
        #  read in montage_fid.save(f"{self.config["DEFAULT"]['subject_dir']}/{self.subject}-montage.fif")

        info = mne.create_info(montage.ch_names, 100, ch_types='eeg')
        raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 100)), info)
        raw.set_montage(montage)

        trans = mne.read_trans(f"{self.config['DEFAULT']['subject_dir']}/{self.subject}-trans.fif")
        eeg, fid, surf = get_aligned_artifacts(raw.info, subject=self.fs_subject, trans=trans,
                                               subjects_dir=fs_subjects_dir, coord_frame='mri',
                                               head_surf=self.head_surf)[:2]

        for label in fid.index.values:
            self.add_acq_sphere(fid, label)
        for label in eeg.index.values:
            self.add_acq_sphere(eeg, label)

    def save_montage_tsv(self):
        df = correct_for_movements(self.df_acq)[["x1", "y1", "z1"]].reset_index()
        df.columns = ["name", "x", "y", "z"]
        montage_fname = f"{self.config['DEFAULT']['subject_dir']}/{self.subject}_electrodes.tsv"
        df.to_csv(montage_fname, sep="\t", index=False)

    def get_acq_montage(self):
        df = correct_for_movements(self.df_acq)[["x1", "y1", "z1"]]
        df_elec = df.drop(index=["NASION", "LPA", "RPA"]).dropna()
        if len(df_elec) == 0:
            return None
        return mne.channels.make_dig_montage(ch_pos=dict(list(zip(df_elec.index.values, df_elec.values))),
                                             nasion=df.loc["NASION"].values,
                                             lpa=df.loc["LPA"].values,
                                             rpa=df.loc["RPA"].values,
                                             coord_frame="head")

    def save_montage_and_trans(self):
        montage_fid = self.get_acq_montage()
        if montage_fid is None:
            print("Return no montage")
            return

        montage_fid.save(f"{self.config['DEFAULT']['subject_dir']}/{self.subject}-montage.fif")
        self.save_montage_tsv()

        df_acq = correct_for_movements(self.df_acq).dropna()
        mri_pts = self.df_montage.loc[df_acq.index.values].values
        mtg_pts = df_acq.values[:, :3]

        trans_mat = recover_homogenous_affine_trans(mtg_pts[:3], mri_pts[:3])
        # rx, ry, rz, tx, ty, tz, sx, sy, sz
        x0 = [*mne.transforms.rotation_angles(trans_mat), *trans_mat[:3, 3], 1.0, 1.0, 1.0]
        n_scale_params = 3
        trans = mne.coreg.fit_matched_points(mtg_pts, mri_pts, x0=x0, out='trans',
                                             scale=n_scale_params) #, weights=(1.0, 10.0, 1.0))
        trans = mne.Transform('head', 'mri', trans)
        trans_file_name = mne.coreg.trans_fname.format(raw_dir=self.config["DEFAULT"]['subject_dir'],
                                                       subject=self.subject)
        mne.write_trans(trans_file_name, trans)


class FastrakWidget(QtWidgets.QSplitter):

    def __init__(self, parent=None):
        super(FastrakWidget, self).__init__(QtCore.Qt.Horizontal, parent)

        self.config = configparser.ConfigParser()

        if Path(config_path).exists():
            self.config.read(config_path)

        self.v_wgt = QtWidgets.QWidget(self)
        v_layout = QtWidgets.QVBoxLayout(self.v_wgt)

        self.input_wgt = CmdInputWidget(self)
        v_layout.addWidget(self.input_wgt)

        self.input_wgt.connect_signal.connect(self.connect_rs232)
        self.input_wgt.disconnect_signal.connect(self.disconnect_rs232)

        self.receive_wgt = CmdReceivedWidget(self)
        v_layout.addWidget(self.receive_wgt)

        self.addWidget(self.v_wgt)

        self.mesh_view = MeshDisplayWidget(self)
        self.addWidget(self.mesh_view)

        self.com_thread = None

        self.input_wgt.connect_rs233()

        self.show()
        self.resize(1200, 800)

    def connection_error(self, message):
        QtWidgets.QMessageBox.warning(self, "Warning", message)
        self.input_wgt.disconnect_rs233()

    def connect_rs232(self):
        if self.com_thread is not None:
            if self.com_thread.isRunning():
                self.disconnect_rs232()

        self.com_thread = ComThread(self.receive_wgt, self.input_wgt,
                                    port=self.input_wgt.rs232_port_wgt.text())
        self.com_thread.command_received.connect(self.mesh_view.command_received)
        self.com_thread.text_read.connect(self.receive_wgt.command_received)
        self.com_thread.connection_failed.connect(self.connection_error)
        self.com_thread.start()

    def disconnect_rs232(self):
        if self.com_thread is not None:
            self.com_thread.continue_running = False
            self.com_thread.wait()
            self.com_thread.command_received.disconnect(self.mesh_view.command_received)
            self.com_thread.text_read.disconnect(self.receive_wgt.command_received)
            self.com_thread = None

    def eventFilter(self, widget, event):

        '''if event.type() == QtCore.QEvent.KeyPress:

            print(event.key())
            # on j
            if event.key() == QtCore.Qt.Key_J:

                print('j')
                return True

            # on k
            elif event.key() == QtCore.Qt.Key_K:

                print('k')
                return True

            # on l - this if statement is never true!
            elif event.key() == QtCore.Qt.Key_L:

                print('l')
                return True
        '''
        return False


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FastrakWidget()
    app.installEventFilter(window)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
