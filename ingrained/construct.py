import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, Delaunay, distance_matrix

# pymatgen tools
from pymatgen.ext.matproj import MPRester
from pymatgen.io.xyz import XYZ
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

class Slab(object):                   
    def __init__(self, chemical_formula, space_group, uvw_project, uvw_upward, tilt_angle, max_dimension): 
        self.chemical_formula = chemical_formula
        self.space_group = space_group
        self.unit_cell = self._query_MP()
        self.slab = self.construct_oriented_slab(uvw_project, uvw_upward, tilt_angle, max_dimension)
        self.width, self.width_tol = self.get_repeat_dist(direction="width")
        self.depth, self.depth_tol = self.get_repeat_dist(direction="depth")

    def _query_MP(self):
        """
        Retrieve a conventional standard unit cell cif from MP

        Args:
            chemical_formula: A string of an element or compound "pretty formula"
        Returns:
            A pymatgen conventional standard unit cell
        """
        mpr = MPRester("MAPI_KEY")
        query = mpr.query(criteria={"pretty_formula": self.chemical_formula}, 
                          properties=["structure","icsd_ids","spacegroup"])
        
        # First filter by space_group if provided 
        if self.space_group:
            query = [query[i] for i in range(len(query)) if 
                     SpacegroupAnalyzer(query[i]['structure']).get_space_group_symbol()==self.space_group]

        # Select minimum volume:
        selected = query[np.argmin([query[i]['structure'].lattice.volume for 
                                                      i in range(len(query))])]

        pymatgen_structure = SpacegroupAnalyzer(selected["structure"]
                                        ).get_conventional_standard_structure()
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

        while a_start < (float(max_dimension+10)*expand)/2:

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
        
        bx_size = ((max_dimension+1)/2)

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

    def filter_atoms(position, select_idx, direction):
        """
        Helper function for "get_repeat_dist"
        
        Args:
            position (list): List of atomic positions
            select_idx (int): selection index for outer loop
            direction (str): Which direction to filter atoms
        
        Returns:
            list of positions
        """
        if direction == 'width':
            # Filter out atoms that have same a
            position_filt = np.array([a for a in position if a[0] != \
                                                      position[select_idx][0]])
            # Filter out atoms that are not at same width (has tolerance)
            position_filt = np.array([a for a in position_filt if np.abs(a[1]-\
                                               position[select_idx][1]) < 1.0])
            # Filter out atoms that are not at same height (has tolerance)
            position_filt = np.array([a for a in position_filt if np.abs(a[2]-\
                                               position[select_idx][2]) < 1.0])
        elif direction == 'depth': 
            # Filter out atoms that have same c
            position_filt = np.array([a for a in position if a[2] != \
                                                      position[select_idx][2]])
            # Filter out atoms that are not at same width (has tolerance)
            position_filt = np.array([a for a in position_filt if np.abs(a[1]-\
                                               position[select_idx][1]) < 0.2])
            # Filter out atoms that are not at same height (has tolerance)
            position_filt = np.array([a for a in position_filt if np.abs(a[0]-\
                                               position[select_idx][0]) < 0.2])

        return(position_filt)
        
        
    def filter_check(position, position_filt, select_idx, direction):
        """
        Helper function for get_repeat_dist
        
        Args:
            position (list): List of atomic positions
            position_filt (list): List of filtered atomic positions
            select_idx (int): selection index for outer loop
            direction (str): Which direction to filter atoms
        
            
        
        """
        
        if position_filt != []:
            if direction=='width':
                # Find index of closest_b (not in column)
                cb_idx = np.argmin(np.abs(position_filt[:,1]-\
                                                      position[select_idx][1]))
                # Compute "a" distance to closest proxy atom
                adis = np.round(np.abs(position[select_idx][0]-\
                                                   position_filt[cb_idx][0]),4)
                # Compute "b" tolerance to closest proxy atom
                btol = np.round(np.abs((position_filt[cb_idx][1]-\
                                                   position[select_idx][1])),4)

            elif direction=='depth':
                # Find index of closest_b (not in column)
                cb_idx = np.argmin(np.abs(position_filt[:,1]-\
                                                      position[select_idx][1]))
                # Compute "a" distance to closest proxy atom
                adis = np.round(np.abs(position[select_idx][2]-\
                                                   position_filt[cb_idx][2]),4)
                # Compute "b" tolerance to closest proxy atom
                btol = np.round(np.abs((position_filt[cb_idx][1]-\
                                                   position[select_idx][1])),4)
 
        return(adis,btol)
        

    def get_repeat_dist(self,direction="width",mode="tol"):
        """
        Find the approximate length needed for one full repeat of the structure
        along width or depth. 

        Args:
            pymatgen_structure: The input pymatgen structure
            direction: The direction along which to find repeat length 
                       (width = perp to uvw_project and uvw_upward, 
                       depth = along uvw_project)
            mode: The decision used to accept the solution (tol = 
                                               min tolerance, len = min length)

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
            indices = [int(i) for i, elem in enumerate(chemical_symbols_list) \
                                                            if element in elem]
            position = np.array([positions_list[idx] for idx in indices])                     
            for select_idx in range(np.shape(position)[0]): 
                position_filt = Slab.filter_atoms(position,select_idx,direction)
                adis, btol = Slab.filter_check(position,
                                          position_filt,
                                          select_idx,
                                          direction)
 
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
                idx = np.argmin(np.array([a[0] for a in dtol_elem if \
                                                             a[1] == min_tol]))
                elem_tols.append(dtol_elem[idx])

        critical_idx = np.argmax(np.array([a[0] for a in elem_tols]))
        return elem_tols[critical_idx][0:2]

    def to_xyz(self,filename):
        xyz = XYZ(self.slab)
        xyz.write_file(filename=filename+".xyz")

    def to_vasp(self,filename):
        self.unit_cell.to(filename=filename.split(".vasp")[0]+".POSCAR.vasp")

    @staticmethod
    def _in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array 
        of the coordinates of `M` points in `K`dimensions for which 
        Delaunay triangulation will be computed
        """
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    @staticmethod
    def _centroid_coordinates(crds):
        return np.sum(crds[:, 0])/crds.shape[0], \
               np.sum(crds[:, 1])/crds.shape[0], \
               np.sum(crds[:, 2])/crds.shape[0]

class TopGrain(Slab):
    def __init__(self, chemical_formula, space_group, uvw_project, uvw_upward, 
                                tilt_angle, max_dimension, flip_species=False):
        super().__init__(chemical_formula, space_group, uvw_project, 
                                         uvw_upward, tilt_angle, max_dimension)
        
        self.height = max_dimension
        self.structure = self.set_in_bicrystal(flip_species=flip_species)

    def set_in_bicrystal(self, flip_species=False):
        """
        Given grains, find the proper width/depth dimensions

        Args:
            pymatgen_structure: The input pymatgen grain structure 
            width: The width of the bicrystal supercell to set grain in 
            height: The height of the bicrystal supercell to set grain in 
            depth: The depth of the bicrystal supercell to set grain in 
            set_in: The position (top/bottom) to set grain in bicrystal 
                    supercell 

        Returns:
            A pymatgen grain structure set into a bicrystal supercell
        """
        atomObj = AseAtomsAdaptor.get_atoms(self.slab)
        height = self.height
        width, depth = self.width, self.depth

        chiseled = atomObj.copy()
        pos = chiseled.get_positions()
        pos -= np.min(pos,axis=0)   
        chiseled.set_positions(pos) 
        inidx = np.all(np.logical_and(pos>=[0,0,0],
                                      pos<[width,height,depth]),
                                      axis=1)
        
        del pos
        del chiseled[np.logical_not(inidx)]

        # Set grain so that min pos[:,1] is at the interface
        pos = chiseled.get_positions()
        pos[:,1] = pos[:,1] - np.min(pos[:,1])+1E-10
        up_shift = ((2*height)- 1E-10)/2
        pos[:,1] = pos[:,1]+up_shift
        chiseled.set_positions(pos) 
        
        # Reset the cell so it is big enough to eventually 
        # accomodate a second grain
        chiseled.set_cell([width,2*height,depth]) 
        
        # Remove structure conflicts at width and depth 
        # boundaries as a result of imperfect PBC
        pmg = AseAtomsAdaptor.get_structure(chiseled)
        s = pmg.copy()
        pos = s.cart_coords
        indel, i = [] , 0
        for crd in pos:
            if crd[0] < 0.5:
                test_crd = crd.copy()
                test_crd[0] += width
                if np.any((distance_matrix([test_crd], pos)[0]< 1) == True) ==\
                                                       True and i not in indel:
                    indel.append(i)
            if crd[2] < 0.5:
                test_crd = crd.copy()
                test_crd[2] += depth
                if np.any((distance_matrix([test_crd], pos)[0]< 1) == True) ==\
                                                       True and i not in indel:
                    indel.append(i)
            i+=1
        s.remove_sites(indel)

        # If flip_species = True and 2 element criteria satisfied
        species_list = [str(a).replace("Element","").strip() for a in \
                                                               list(s.species)]
        unique_species = list(set(species_list))
        if len(unique_species) == 2 and flip_species:
            for i in range(len(s)):
                s[i] = unique_species[int(str(s[i].specie)==unique_species[0])]
        return s

    def to_vasp(self,filename):
        self.structure.to(filename=filename+".POSCAR.vasp")

    def to_xyz(self,filename):
        XYZ(self.structure).write_file(filename=filename+".xyz")

class BottomGrain(Slab):
    def __init__(self, chemical_formula, space_group, uvw_project, 
                     uvw_upward, tilt_angle, max_dimension,flip_species=False):
        super().__init__(chemical_formula, space_group, uvw_project, 
                                         uvw_upward, tilt_angle, max_dimension)
        
        self.height = max_dimension
        self.structure = self.set_in_bicrystal(flip_species=flip_species)

    def set_in_bicrystal(self,flip_species=False):
        """
        Given grains, find the proper width/depth dimensions

        Args:
            pymatgen_structure: The input pymatgen grain structure 
            width: The width of the bicrystal supercell to set grain in 
            height: The height of the bicrystal supercell to set grain in 
            depth: The depth of the bicrystal supercell to set grain in 
            set_in: The position (top/bottom) to set grain in bicrystal 
            supercell 

        Returns:
            A pymatgen grain structure set into a bicrystal supercell
        """
        atomObj = AseAtomsAdaptor.get_atoms(self.slab)
        height = self.height
        width, depth = self.width, self.depth

        chiseled = atomObj.copy()
        pos = chiseled.get_positions()
        pos -= np.min(pos,axis=0)   
        chiseled.set_positions(pos) 
        inidx = np.all(np.logical_and(pos>=[0,0,0],
                                      pos<[width,height,depth]),
                                      axis=1)
        
        del pos
        del chiseled[np.logical_not(inidx)]
        
        # Set grain so that max pos[:,1] is at the interface
        pos = chiseled.get_positions()
        pos[:,1] = pos[:,1] - np.min(pos[:,1])+1E-10
        up_shift = ((2*height)- 1E-10)/2 - np.max(pos[:,1])
        pos[:,1] = pos[:,1]+up_shift
        chiseled.set_positions(pos) 

        # Reset the cell so it is big enough to eventually accomodate a second grain
        chiseled.set_cell([width,2*height,depth]) 
        
        # Remove structure conflicts at width and depth boundaries as a result of imperfect PBC
        pmg = AseAtomsAdaptor.get_structure(chiseled)
        s = pmg.copy()
        pos = s.cart_coords
        indel, i = [] , 0
        for crd in pos:
            if crd[0] < 0.5:
                test_crd = crd.copy()
                test_crd[0] += width
                if np.any((distance_matrix([test_crd], pos)[0]< 1) == True) ==\
                                                       True and i not in indel:
                    indel.append(i)
            if crd[2] < 0.5:
                test_crd = crd.copy()
                test_crd[2] += depth
                if np.any((distance_matrix([test_crd], pos)[0]< 1) == True) ==\
                                                       True and i not in indel:
                    indel.append(i)
            i+=1
        s.remove_sites(indel)

        # If flip_species = True and 2 element criteria satisfied
        species_list = [str(a).replace("Element","").strip() for a in \
                                                               list(s.species)]
        unique_species = list(set(species_list))
        if len(unique_species) == 2 and flip_species:
            for i in range(len(s)):
                s[i] = unique_species[int(str(s[i].specie)==unique_species[0])]
        return s

    def to_vasp(self,filename):
        self.structure.to(filename=filename+".POSCAR.vasp")

    def to_xyz(self,filename):
        XYZ(self.structure).write_file(filename=filename+".xyz")
