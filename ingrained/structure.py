import os
import cv2
import sys
import imutils
import subprocess
import numpy as np
import itertools as it
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
from scipy.spatial import distance

import dm3_lib as dm3lib
import matplotlib.pyplot as plt

# You will need your own key!
os.environ['MAPI_KEY'] = "¯\_(ツ)_/¯"
# You will need your own key!

from pymatgen import MPRester
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar, Structure, Lattice
from pymatgen.io.xyz import XYZ

from pymatgen.io.vasp.outputs import Chgcar

# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass
class BorderValueError(Error):
   """Raised when reduction of image border is negative or >45%"""
   pass
class PixelSizeError(Error):
   """Raised when the input value is too large"""
   pass
class OSExecutableError(Error):
   """Raised when the input value is too large"""
   pass

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class Slab(object):                   
    def __init__(self, chemical_formula, space_group, uvw_project, uvw_upward, tilt_angle, max_dimension): 
        self.chemical_formula = chemical_formula
        self.space_group = space_group
        self.unit_cell = self._query_MP()
        self.slab = self.construct_oriented_slab(uvw_project, uvw_upward, tilt_angle, max_dimension)
        self.width_val, self.width_tol = self.get_repeat_dist(direction="width")
        self.depth_val, self.depth_tol = self.get_repeat_dist(direction="depth")

    def _query_MP(self):
        """
        Retrieve a conventional standard unit cell cif from MP

        Args:
            chemical_formula: A string of an element or compound "pretty formula"
        Returns:
            A pymatgen conventional standard unit cell
        """
        mpr = MPRester()
        query = mpr.query(criteria={"pretty_formula": self.chemical_formula}, properties=["structure","icsd_ids"])
        
        # First filter by space_group if provided 
        if self.space_group:
            query = [query[i] for i in range(len(query)) if SpacegroupAnalyzer(query[i]['structure']).get_space_group_symbol()==self.space_group]

        # Select minimum volume:
        selected = query[np.argmin([query[i]['structure'].lattice.volume for i in range(len(query))])]

        pymatgen_structure = SpacegroupAnalyzer(selected["structure"]).get_conventional_standard_structure()
        # pymatgen_structure.to(filename=self.chemical_formula+'_unit.POSCAR.vasp')
        return pymatgen_structure

    def _construct_slab(self,max_dimension):
        """
        Construct a supercell slab from a conventional standard unit cell
        (i.e. a larger chunk of the material intended for further shaping or finishing)

        Args:
            pymatgen_structure: The input pymatgen conventional standard unit cell
            max_dimension: A float representing the max edge length of the supercell
        Returns:
            A pymatgen supercell (max_dimension preserved when structure cubed after rotation)
        """
        # 
        supercell = self.unit_cell.copy()

        # Each row should correspond to a lattice vector.
        lattice_matrix = self.unit_cell.lattice.matrix
        bounding_box = np.vstack([lattice_matrix,np.matmul([[0,0,0],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],lattice_matrix)])

        # Starting edge length in Angstroms (cube)
        a_start = 0.5
        expansion_vector = np.array([1,1,1])
        expansion_matrix = lattice_matrix
        expand = np.sqrt(3)# [value_false, value_true][<test>] 

        while a_start < (float(max_dimension)*expand)/2:

            # Find lattice vector with minimum length   
            _ , idx = KDTree(expansion_matrix).query([0,0,0])
            
            # Update expansion vector
            expansion_vector[idx]+= 1
            
            # Update expansion matrix 
            expansion_matrix = (lattice_matrix.T*expansion_vector).T

            # Update bounding box
            bounding_box = np.vstack([expansion_matrix,np.matmul([[0,0,0],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],expansion_matrix)])

            # Center bounding box to (0,0,0)
            bounding_box = bounding_box - self._centroid_coordinates(bounding_box)

            # Expand cube from center until penetrates bounding box
            for a in np.arange(a_start,a_start+50,0.5):
                sample = np.array(list(it.product((-a, a), (-a, a), (-a, a))))
                inside =  all(self._in_hull(sample,bounding_box))
                if not inside:
                    a_start = a
                    break

        supercell.make_supercell(expansion_vector)
        return supercell

    def construct_oriented_slab(self,uvw_project,uvw_upward,tilt_angle,max_dimension):
        """ 
        Construct a slab, oriented for a specific direction of viewing, and slice into a cube

        Args:
            pymatgen_structure: The input pymatgen structure
            uvw_project: The direction of projection (sceen to viewer) is a lattice vector ua + vb + wc.
            uvw_upward: The upward direction is a lattice vector ua + vb + wc (must be normal to along_uvw).
            tilt_ang: The CCW rotation around 'uvw_project' (applied after uvw_upward is set)
            max_dimension: A float representing the max edge length of the supercell

        Returns:
            A pymatgen supercell (cube) oriented for a specific direction of viewing 
        """
        slab = self._construct_slab(max_dimension)

        atomObj = AseAtomsAdaptor.get_atoms(slab)

        # Make copy of atom object and get cell/projection vector info
        atom  = atomObj.copy()

        # Convert u,v,w vector to cartesian
        along_xyz = np.matmul(np.array(atom.get_cell()).T,uvw_project)

        # Rotate coordinates and cell so that 'along_xyz' is coincident with [0,0,1] 
        atom.rotate(along_xyz,[0,0,1],rotate_cell=True)

        # Convert u,v,w vector to cartesian
        upwrd_xyz = np.matmul(np.array(atom.get_cell()).T,uvw_upward)

        # Rotate coordinates and cell so that 'upwrd_xyz' is coincident with [0,1,0] 
        atom.rotate(upwrd_xyz,[0,1,0],rotate_cell=True)

        # Rotate coordinates along z to capture tilt angle
        atom.rotate(tilt_angle,'z')
        
        bx_size = (max_dimension/2)

        pos = atom.get_positions()
        pos -= self._centroid_coordinates(atom.get_positions())    
        atom.set_positions(pos) 

        inidx = np.all(np.logical_and(pos>=[-bx_size,-bx_size,-bx_size],pos<=[bx_size,bx_size,bx_size]),axis=1)

        if not np.sum(inidx)>0:
            warnings.warn('Unit cell entirely outside bounding volume')
          
        del atom[np.logical_not(inidx)]

        # Enforce (0,0,0) origin and reset cell around rotated/chiseled slab
        pos = atom.get_positions()
        pos -= np.min(atom.get_positions(),axis=0)
        atom.set_positions(pos)
        atom.set_cell(np.max(atom.get_positions(),axis=0) * np.identity(3))

        return AseAtomsAdaptor.get_structure(atom)

    def get_repeat_dist(self,direction="width",mode="tol"):
        """
        Find the approximate length needed for one full repeat of the structure along width or depth. 

        Args:
            pymatgen_structure: The input pymatgen structure
            direction: The direction along which to find repeat length 
                       (width = perp to uvw_project and uvw_upward, depth = along uvw_project)
            mode: The decision used to accept the solution (tol = min tolerance, len = min length)

        Returns:
            A float representing the length needed for structure to repeat. 
        """
        slab = AseAtomsAdaptor.get_atoms(self.slab)
        positions_list = slab.get_positions()
        chemical_symbols_list = slab.get_chemical_symbols()
        chemical_symbols = list(set(chemical_symbols_list))

        dtol = []
        for element in chemical_symbols:
            # Get all indices of 
            indices = [int(i) for i, elem in enumerate(chemical_symbols_list) if element in elem]
            position = np.array([positions_list[idx] for idx in indices])                     
            for select_idx in range(np.shape(position)[0]):           
                if direction == 'width':
                    # Filter out atoms that have same a
                    position_filt = np.array([a for a in position if a[0] != position[select_idx][0]])
                    # Filter out atoms that are not at same width (has tolerance)
                    position_filt = np.array([a for a in position_filt if np.abs(a[1] - position[select_idx][1]) < 1.0])
                    # Filter out atoms that are not at same height (has tolerance)
                    position_filt = np.array([a for a in position_filt if np.abs(a[2] - position[select_idx][2]) < 1.0])
                    if position_filt != []:
                        # Find index of closest_b (not in column)
                        cb_idx = np.argmin(np.abs(position_filt[:,1]-position[select_idx][1]))
                        # Compute "a" distance to closest proxy atom
                        adis = np.round(np.abs(position[select_idx][0]-position_filt[cb_idx][0]),4)
                        # Compute "b" tolerance to closest proxy atom
                        btol = np.round(np.abs((position_filt[cb_idx][1]-position[select_idx][1])),4)
                elif direction == 'depth': 
                    # Filter out atoms that have same c
                    position_filt = np.array([a for a in position if a[2] != position[select_idx][2]])
                    # Filter out atoms that are not at same width (has tolerance)
                    position_filt = np.array([a for a in position_filt if np.abs(a[1] - position[select_idx][1]) < 0.2])
                    # Filter out atoms that are not at same height (has tolerance)
                    position_filt = np.array([a for a in position_filt if np.abs(a[0] - position[select_idx][0]) < 0.2])
                    if position_filt != []:
                        # Find index of closest_b (not in column)
                        cb_idx = np.argmin(np.abs(position_filt[:,1]-position[select_idx][1]))
                        # Compute "a" distance to closest proxy atom
                        adis = np.round(np.abs(position[select_idx][2]-position_filt[cb_idx][2]),4)
                        # Compute "b" tolerance to closest proxy atom
                        btol = np.round(np.abs((position_filt[cb_idx][1]-position[select_idx][1])),4)
                try:
                    if (adis,btol) not in dtol:
                        dtol.append((adis,btol,element))
                except:
                    pass

        if mode == "tol" :
            elem_tols = []
            for element in chemical_symbols:
                dtol_elem = [a for a in dtol if a[2] == element]
                min_tol = np.min(np.array([a[1] for a in dtol_elem]))
                idx = np.argmin(np.array([a[0] for a in dtol_elem if a[1] == min_tol]))
                elem_tols.append(dtol_elem[idx])

        critical_idx = np.argmax(np.array([a[0] for a in elem_tols]))
        return elem_tols[critical_idx][0:2]

    def to_xyz(self,filename):
        xyz = XYZ(self.slab)
        xyz.write_file(filename=filename.split(".xyz")[0]+".xyz")

    def to_vasp(self,filename):
        self.unit_cell.to(filename=filename.split(".vasp")[0]+".POSCAR.vasp")

    @staticmethod
    def _in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    @staticmethod
    def _centroid_coordinates(crds):
        return np.sum(crds[:, 0])/crds.shape[0], np.sum(crds[:, 1])/crds.shape[0], np.sum(crds[:, 2])/crds.shape[0]

class TopGrain(Slab):
    def __init__(self, chemical_formula, space_group, uvw_project, uvw_upward, tilt_angle, max_dimension, grain_height=""):
        super().__init__(chemical_formula, space_group, uvw_project, uvw_upward, tilt_angle, max_dimension)
        if grain_height == "":
            self.height_val = max_dimension
        else:
            self.height_val = grain_height

        self.structure = self.set_in_bicrystal(self.height_val,'top')

    def set_in_bicrystal(self,height,set_in):
        """
        Given grains, find the proper width/depth dimensions

        Args:
            pymatgen_structure: The input pymatgen grain structure 
            width: The width of the bicrystal supercell to set grain in 
            height: The height of the bicrystal supercell to set grain in 
            depth: The depth of the bicrystal supercell to set grain in 
            set_in: The position (top/bottom) to set grain in bicrystal supercell 

        Returns:
            A pymatgen grain structure set into a bicrystal supercell
        """
        atomObj = AseAtomsAdaptor.get_atoms(self.slab)

        width, depth = self.width_val, self.depth_val

        chiseled = atomObj.copy()
        pos = chiseled.get_positions()
        pos -= np.min(pos,axis=0)   
        chiseled.set_positions(pos) 
        inidx = np.all(np.logical_and(pos>=[0,0,0],pos<[width,height,depth]),axis=1)
        
        del pos
        del chiseled[np.logical_not(inidx)]

        if set_in == 'top':
            pos = chiseled.get_positions()
            pos[:,1] = pos[:,1]+height
            chiseled.set_positions(pos) 
            
        chiseled.set_cell([width,2*height,depth]) 

        # return chiseled
        return AseAtomsAdaptor.get_structure(chiseled)

    def to_vasp(self,filename):
        self.structure.to(filename=filename.split(".vasp")[0]+".POSCAR.vasp")

class BottomGrain(Slab):
    def __init__(self, chemical_formula, space_group, uvw_project, uvw_upward, tilt_angle, max_dimension, grain_height=""):
        super().__init__(chemical_formula, space_group, uvw_project, uvw_upward, tilt_angle, max_dimension)
        if grain_height == "":
            self.height_val = max_dimension
        else:
            self.height_val = grain_height

        self.structure = self.set_in_bicrystal(self.height_val,'bottom')

    def set_in_bicrystal(self,height,set_in):
        """
        Given grains, find the proper width/depth dimensions

        Args:
            pymatgen_structure: The input pymatgen grain structure 
            width: The width of the bicrystal supercell to set grain in 
            height: The height of the bicrystal supercell to set grain in 
            depth: The depth of the bicrystal supercell to set grain in 
            set_in: The position (top/bottom) to set grain in bicrystal supercell 

        Returns:
            A pymatgen grain structure set into a bicrystal supercell
        """
        atomObj = AseAtomsAdaptor.get_atoms(self.slab)

        width, depth = self.width_val, self.depth_val

        chiseled = atomObj.copy()
        pos = chiseled.get_positions()
        pos -= np.min(pos,axis=0)   
        chiseled.set_positions(pos) 
        inidx = np.all(np.logical_and(pos>=[0,0,0],pos<[width,height,depth]),axis=1)
        
        del pos
        del chiseled[np.logical_not(inidx)]

        if set_in == 'top':
            pos = chiseled.get_positions()
            pos[:,1] = pos[:,1]+height
            chiseled.set_positions(pos) 
            
        chiseled.set_cell([width,2*height,depth]) 

        # return chiseled
        return AseAtomsAdaptor.get_structure(chiseled)

    def to_vasp(self,filename):
        self.structure.to(filename=filename.split(".vasp")[0]+".POSCAR.vasp")

class Bicrystal(object):
    def __init__(self, structure="", minmax_width="", minmax_depth=""):
        if isinstance(structure,list):
            grain_1, grain_2 = structure
            self.top_grain = TopGrain(grain_1['chemical_formula'],grain_1['space_group'],\
                                      grain_1['uvw_project'], grain_1['uvw_upward'],\
                                      grain_1['tilt_angle'],  grain_1['max_dimension'])
            self.bot_grain = BottomGrain(grain_2['chemical_formula'],grain_2['space_group'],\
                                      grain_2['uvw_project'], grain_2['uvw_upward'],\
                                      grain_2['tilt_angle'],  grain_2['max_dimension'])
            self.top_grain.strain, self.bot_grain.strain = self._strain_to_coincidence(minmax_width=minmax_width,minmax_depth=minmax_depth)
            self.structure = self._fuse_grains()
            self.haadf_image = None
            self.haadf_pixel = None
        else:
            self.top_grain = None 
            self.bot_grain = None
            self.structure = Poscar.from_file(structure).structure
            self.haadf_image = None
            self.haadf_pixel = None

    def _strain_to_coincidence(self, minmax_width="", minmax_depth=""):
        """
        Take the current grain templates, expand in width/depth to minimize strain 
        between grains and assign same width/depth computed as the average of the 
        ideal expansion values. 

        Returns:
            Tuples of (width,depth) strain values for top and bottom (+ updates grain.structures)
        """
        grain_1_width = self.top_grain.width_val
        grain_1_depth = self.top_grain.depth_val

        grain_2_width = self.bot_grain.width_val
        grain_2_depth = self.bot_grain.depth_val

        if not minmax_width:
            # The bicrystal cannot be any smaller in width than than the min width of either constituent grain
            min_width = min([grain_1_width,grain_2_width])
            # The bicrystal cannot be any larger in width than 1.1 x the height of a single grain
            max_width = self.top_grain.height_val*1.1
            minmax_width = (min_width,max_width)

        if not minmax_depth:
            # The bicrystal cannot be any smaller in depth than the min depth of either constituent grain
            min_depth = min([grain_1_depth,grain_2_depth])
            # The bicrystal cannot be any larger in depth than 3 x max depth of either constituent grain
            max_depth = 3*max([grain_1_depth,grain_2_depth])
            minmax_depth = (min_depth,max_depth)

        top_width, bot_width, tol_width = self._find_approx_lcm(grain_1_width,grain_2_width,min_len=minmax_width[0],max_len=minmax_width[1])
        top_depth, bot_depth, tol_depth = self._find_approx_lcm(grain_1_depth,grain_2_depth,min_len=minmax_depth[0],max_len=minmax_depth[1])

        top_expansion = [int(top_width/grain_1_width),1,int(top_depth/grain_1_depth)]
        bot_expansion = [int(bot_width/grain_2_width),1,int(bot_depth/grain_2_depth)]

        self.top_grain.structure.make_supercell(top_expansion)
        self.bot_grain.structure.make_supercell(bot_expansion)

        widths = [self.top_grain.structure.lattice.abc[0],self.bot_grain.structure.lattice.abc[0]]
        depths = [self.top_grain.structure.lattice.abc[2],self.bot_grain.structure.lattice.abc[2]]

        # Average for top/bottom widths and depths
        avg_width = np.mean(widths)
        avg_depth = np.mean(depths)

        # Details on tension/compression
        strain_top_width = ((avg_width-widths[0])/widths[0])*100
        strain_bot_width = ((avg_width-widths[1])/widths[1])*100
        strain_top_depth = ((avg_depth-depths[0])/depths[0])*100
        strain_bot_depth = ((avg_depth-depths[1])/depths[1])*100

        # print("\n")
        # print("-"*36)
        # print("Strain in top grain (%): ")
        # print("  >> width : ",strain_top_width)
        # print("  >> depth : ",strain_top_depth)
        # print("Strain in bottom grain (%): ")
        # print("  >> width : ",strain_bot_width)
        # print("  >> depth : ",strain_bot_depth)
        # print("-"*36)
        # print("\n")

        # Expand/contract each grain to coincidence in 'a' and 'c' direction
        # Note: ASE has nice functionality here - not sure if pymatgen has same function?
        top_grain = AseAtomsAdaptor.get_atoms(self.top_grain.structure)
        bot_grain = AseAtomsAdaptor.get_atoms(self.bot_grain.structure)
        
        top_grain.set_cell([np.mean(widths),self.top_grain.height_val*2,np.mean(depths)],scale_atoms=True) 
        bot_grain.set_cell([np.mean(widths),self.top_grain.height_val*2,np.mean(depths)],scale_atoms=True) 

        self.top_grain.structure = AseAtomsAdaptor.get_structure(top_grain)
        self.bot_grain.structure = AseAtomsAdaptor.get_structure(bot_grain)

        return (strain_top_width,strain_top_depth), (strain_bot_width,strain_bot_depth)

    def _fuse_grains(self):
        """
        Combine two bicrystal supercell into a single bicrystal supercell

        Args:
            pymatgen_structure_1: The input top bicrystal pymatgen structure
            pymatgen_structure_2: The input bot bicrystal pymatgen structure

        Returns:
            Tuples of top/bot widths and depths as floats
        """
        top_grain = AseAtomsAdaptor.get_atoms(self.top_grain.structure)
        bot_grain = AseAtomsAdaptor.get_atoms(self.bot_grain.structure)

        top = top_grain.copy()
        bot = bot_grain.copy()

        chem_sym_list = top.get_chemical_symbols()

        j=0
        for top_crds in top.get_positions():
            bot.append(Atom(chem_sym_list[j],top_crds))
            j+=1

        gb_pos = bot.get_positions()
        gb_cel = bot.get_cell_lengths_and_angles()[0:3]
        accepted_pos = np.vstack([np.vstack([gb_pos[:,2],gb_pos[:,0]]),gb_pos[:,1]]).T

        bot.set_positions(accepted_pos) 
        bot.set_cell([gb_cel[2],gb_cel[0],gb_cel[1]])
       
        return AseAtomsAdaptor.get_structure(bot)

    def _adjust_interface_width(self,width):

        if width == 0:
            return self.structure.copy()

        bc = self.structure.copy()
 
        newa = bc.lattice.a
        newb = bc.lattice.b
        newc = bc.lattice.c+width

        new_crds = []
        new_spec = []
        for idx in range(len(bc)):
            cx,cy,cz = bc.cart_coords[idx]
            if bc.cart_coords[idx][2] >= bc.lattice.c/2:
                new_crds.append([cx/newa,cy/newb,(cz+(1.5*width))/newc])
            else:
                new_crds.append([cx/newa,cy/newb,(cz+(0.5*width))/newc])
            
            new_spec.append(str(bc.species[idx]).split('Element')[0])
        
        lattice = Lattice.from_parameters(a=newa, b=newb, c=newc, alpha=90, beta=90, gamma=90)
        return Structure(lattice, new_spec, new_crds)

    def convolution_HAADF(self, filename="", dm3="", pixel_size="", interface_width=0, defocus=1.0, border_reduce=(0,0)):
        """
        filename: A string representing the file name. The filename must include image format like .jpg, .png,

        """
        try: 
            if(border_reduce[0] >= 0.20 or border_reduce[0] < 0 or \
               border_reduce[1] >= 0.25 or border_reduce[1] < 0):
                raise BorderValueError

            if not isinstance(pixel_size,str):
                if pixel_size >= 0.8 or pixel_size<0.005:
                    raise PixelSizeError

            if sys.platform == 'darwin':
                opsys = 'mac'
            elif sys.platform == 'linux':
                opsys = 'linux'
            else:
                raise OSExecutableError

        except BorderValueError:
            print("Error: Unstable border reduction!")
            return None
        except PixelSizeError:
            print("Error: Image pixel size (height/width) must be nonegative and < 0.8 Å!")
            return None
        except OSExecutableError:
            print("Error: 'incostem' incompatible with {} operating system!".format(sys.platform))
            return None

        bc = self._adjust_interface_width(interface_width)

        # Get cell parameters 
        atomcel = bc.lattice.abc

        # Get atom positions 
        atompos = bc.cart_coords

        # Set pixel size automatically if not provided
        if not pixel_size:
            dm3fObj = dm3lib.DM3(dm3)
            if dm3fObj.pxsize[1].decode('UTF-8') == 'nm':
                pixel_size = dm3fObj.pxsize[0]*10
            else:
                print("Imaging unit unrecognized! Default 'pixel_size=0.15' in angstroms")
                pixel_size = 0.15

        # print("Pixel size: {} angstroms".format(pixel_size))

        # Compute the simulated image size to enforce consistent scaling
        pixx, pixy = [int(np.round(a/pixel_size)) for a in atomcel[1::]]

        total_atoms = 0
        with open(os.path.dirname(__file__)+'/simulation/SAMPLE.XYZ', "w") as sf:
            sf.write('Kirkland incostem input format\n')
            sf.write(" "*5+"".join(str(format(word, '8.4f')).ljust(10) for word in [atomcel[1],atomcel[2],atomcel[0]])+"\n")
            for idx in range(len(atompos)):
                atom_position = [atompos.tolist()[idx][1],atomcel[2]-atompos.tolist()[idx][2],atompos.tolist()[idx][0]]
                # Atoms on boundary get positions at 0 and at boundary - do NOT print one on boundary to avoid pesky overlap effects.
                diff = np.array([atomcel[1],atomcel[2],atomcel[0]])-np.array(atom_position)
                if not np.any(diff<0.01):
                    sf.write(" "*2+str(bc.atomic_numbers[idx])+" "+\
                          "".join(str(format(word, '8.4f')).ljust(10) for word in [atom_position[0],atom_position[1],atom_position[2]])+\
                          " "+"1.0"+" "*3+"0.076\n")
                    total_atoms+=1
            sf.write(" "*2+"-1")

        # For debuggin purposes - we can see the structure that is created for the image simulation input file!
        # with open(os.path.dirname(__file__)+'/simulation/demo.xyz', "w") as cf:
        #     cf.write(str(total_atoms)+"\n")
        #     cf.write("demo xyz structure file\n")
        #     for idx in range(len(atompos)):
        #         atom_position = [atompos.tolist()[idx][1],atomcel[2]-atompos.tolist()[idx][2],atompos.tolist()[idx][0]]
        #         # Atoms on boundary get positions at 0 and at boundary - do NOT print one on boundary to avoid pesky overlap effects.
        #         diff = np.array([atomcel[1],atomcel[2],atomcel[0]])-np.array(atom_position)
        #         if not np.any(diff<0.01):
        #             cf.write(str(bc.species[idx])+" "+"".join(str(format(word, '8.6f')).ljust(10) for word in [atom_position[0],atom_position[1],atom_position[2]])+"\n")

        with open(os.path.dirname(__file__)+'/simulation/params.txt', "w") as pf:
            pf.write('SAMPLE.XYZ\n1 1 1\nSAMPLE.TIF\n'+str(pixx)+" "+str(pixy)+"\n")            
            pf.write("200 0 0 0 30\n100 150\nEND\n"+str(defocus)+"\n0")             

        with cd(os.path.dirname(__file__)+'/simulation'):
            subprocess.call("./incostem-"+opsys, stdout=subprocess.PIPE)
     
        im = np.asarray(cv2.imread(os.path.dirname(__file__)+'/simulation/SAMPLE.TIF',0))

        xbuf, ybuf = [int(border_reduce[0]*np.shape(im)[1]),int(border_reduce[1]*np.shape(im)[0])]
        im = im[ybuf:np.shape(im)[0]-ybuf,xbuf:np.shape(im)[1]-xbuf]

        if filename:
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(os.path.splitext(filename)[0]+".jpg",im)
        
        os.remove(os.path.dirname(__file__)+'/simulation/SAMPLE.XYZ')
        os.remove(os.path.dirname(__file__)+'/simulation/SAMPLE.TIF')
        os.remove(os.path.dirname(__file__)+'/simulation/params.txt')

        self.haadf_image = im
        self.haadf_pixel = pixel_size

        return im

    def to_vasp(self,filename,interface_width=0):

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if interface_width:
            new_struct = self._adjust_interface_width(interface_width)
        else:
            new_struct = self.structure

        new_struct.to(filename=filename.split(".vasp")[0]+".POSCAR.vasp")

    @staticmethod
    def _find_approx_lcm(a,b,min_len=5,max_len=100):
        """
        Given two floats, a and b, return the smallest floats a' and b' that approximate a multiple of both.

        Args:
            a: value 1
            b: value 2
            min_length: min value of an approximate multiple
            max_length: max value of an approximate multiple

        Returns:
            Approximate smallest multiples a', b', with tolerance
        """
        alist = [a*i for i in range(1,100) if a*i < max_len and a*i > min_len]
        blist = [b*i for i in range(1,100) if b*i < max_len and b*i > min_len]
        dmat = distance.cdist(np.array(alist).reshape(-1,1),np.array(blist).reshape(-1,1),'euclidean')
        result = min((min((v, c) for c, v in enumerate(row)), r) for r, row in enumerate(dmat))
        return alist[result[1]], blist[result[0][1]], np.min(dmat)


class PartialCharge(object):
    def __init__(self, parchg_file="", minmax_width="", minmax_depth=""):
        self.object = Chgcar.from_file(parchg_file)
        self.structure = self.object.structure
        self.stm_image = None

    def stm(self, filename="", dm3="", pixel_size="", rotation_angle="", zthick="", ztol="", rho0="", rho_tol=""):
        """
        filename: A string representing the file name. The filename must include image format like .jpg, .png,
        zthick - (float) thickness or depth from top in Angstroms
        ztol - (float) Distance above the surface to consider
        rho0 - (float) isosurface charge density plane
        rho_tol - (float) tolerance to consider while determining isosurface

        Credit: Chaitanya Kolluru
        """
        chg_obj = self.object
        structure = self.structure
        dim = chg_obj.dim

        chg_data = chg_obj.data['total']
        z_carts = structure.cart_coords[:, 2]
        slab_thickness = z_carts.max() - z_carts.min()
        vol = structure.volume
        rho_max = (chg_data/vol).max()
        zmax = z_carts.max() / structure.lattice.c

        Px_x = structure.lattice.a / dim[0] 
        Px_y = structure.lattice.b / dim[1]

        ######## Check for invalid parameters ########
        if not (-1 < zthick < 0.5*slab_thickness): # min thickness of 1 Ang assumed
            print ("Exception: z thickness is greater than slab thickness")
            return None
        else: # convert distances to fractional coordinates
            zthick = zthick / structure.lattice.c
            ztol = ztol / structure.lattice.c

        if not zmax+ztol < 1:
            print ("Exception: zmax+ztol exceeds 1.0!")
            return None

        if not (rho_max/3 < rho0 < rho_max):
            print ("Exception: rho0 {} is out of bounds".format(rho0))
            return None

        if not (0.0001 < rho_tol < 0.975*rho0):
        # if not (0.0001 < rho_tol < 0.99*rho0):
        # if not (0.0001 < rho_tol < rho0):
            print("Exception: rho_tol {} out-of-bounds!".format(rho_tol))
            return None

        if not pixel_size:
            pixel_size = np.min([Px_x,Px_y])

        if not (0.145 < pixel_size < 0.185):
            print("Exception: pixel size {} out-of-bounds!".format(pixel_size))
            return None
        ##############################################

        nzmin = int(dim[2] * (zmax - zthick))
        nzmax = int(dim[2] * (zmax + ztol))

        try:
            X, Y, Z = [], [], []
            for x in range(dim[0]):
                for y in range(dim[1]):
                    for z in range(nzmin, nzmax):
                        rho = chg_data[x][y][z]/vol
                        if abs(rho - rho0) < rho_tol:
                            X.append(x)
                            Y.append(y)
                            Z.append(z)
            points_3d = []
            for i, j, k in zip(X, Y, Z):
                points_3d.append((i, j, k))

            xy_dict = {}
            for p_3d in points_3d:
                p_xy = p_3d[:2]
                p_z = p_3d[2]
                if p_xy in xy_dict:
                    if xy_dict[p_xy] < p_z:
                        xy_dict[p_xy] = p_z
                else:
                    xy_dict[p_xy] = p_z

            min_z = min(list(xy_dict.values()))
            grid = np.zeros((dim[0], dim[1])) + min_z - 5
            keys = xy_dict.keys()
            for key in keys:
                xi = key[0]
                yi = key[1]
                zi = xy_dict[key]
                grid[xi][yi] = zi

            img_tile = grid * (structure.lattice.c / dim[2])
        except:
            return None

        if not pixel_size:
            Px_x = structure.lattice.a / dim[0] 
            Px_y = structure.lattice.b / dim[1]
            pixel_size = np.min([Px_x,Px_y])


        dim = (int(np.round(structure.lattice.abc[1]/float(pixel_size))),\
               int(np.round(structure.lattice.abc[0]/float(pixel_size))))
        img_tile = cv2.resize(img_tile,dim, interpolation=cv2.INTER_AREA)
        img_tile = np.tile(img_tile,(9,9))

        rotated = imutils.rotate(img_tile, rotation_angle)
        edge_length = int(np.floor((np.min(np.shape(img_tile))/2)*np.sqrt(2)))
        im = self._crop_center(rotated,edge_length,edge_length)

        if filename:
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            img = im.astype(np.float64)
            scaled = (img - img.min()) / (img.max() - img.min())
            scaled = (255*scaled).astype(np.uint8)
            cv2.imwrite(os.path.splitext(filename)[0]+".jpg",scaled)

        self.stm_image = im
        return im

    @staticmethod
    def _crop_center(img,cropx,cropy):
        y,x = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2    
        return img[starty:starty+cropy, startx:startx+cropx]
