from rdkit import Chem
import subprocess
import os
from vina import Vina


def get_size(mol, mol3d, addition=0):
    #配体口袋盒子大小
    max_x = max(mol3d.GetAtomPosition(idx).x for idx in range(len(mol.GetAtoms())))
    min_x = min(mol3d.GetAtomPosition(idx).x for idx in range(len(mol.GetAtoms())))
    size_x = max_x - min_x + addition

    max_y = max(mol3d.GetAtomPosition(idx).y for idx in range(len(mol.GetAtoms())))
    min_y = min(mol3d.GetAtomPosition(idx).y for idx in range(len(mol.GetAtoms())))
    size_y = max_y - min_y + addition

    max_z = max(mol3d.GetAtomPosition(idx).z for idx in range(len(mol.GetAtoms())))
    min_z = min(mol3d.GetAtomPosition(idx).z for idx in range(len(mol.GetAtoms())))
    size_z = max_z - min_z + addition
    coord = [size_x,size_y,size_z]
    return coord

def get_center(mol,mol3d):
    #配体结合位置的中心坐标
    coord=[0,0,0]
    num=0
    x=0
    y=0
    z=0
    for ind,at in enumerate(mol.GetAtoms()):
        num+=1
        x+=mol3d.GetAtomPosition(ind).x
#        print("x",x)
        y+=mol3d.GetAtomPosition(ind).y
        z+=mol3d.GetAtomPosition(ind).z
#        print(x,y,z,num)
    x=x/num
    y=y/num
    z=z/num
    coord=[x,y,z]
    return coord

if __name__ == '__main__':
    #输入数据路径
    data_path = 'data//to_predict'
    #equibind的结果路径
    result_path = 'data//results//output'
    #vina的输出路径
    output_path = 'data//results_vina'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #equibind的输出的名字都统一为'lig_equibind_corrected'
    input_lig_name = 'lig_equibind_corrected'
    names = sorted(os.listdir(data_path))

    for name in names:
        #蛋白质数据在<name>文件夹下
        rec_name = name+'_protein_processed'
        #获得配体和受体信息
        lig_input_file = os.path.join(result_path, name, f'{input_lig_name}.sdf')
        rec_input_file = os.path.join(data_path,name, f'{rec_name}.pdb')
        #用rdkit处理获得docking位置
        mols_suppl = Chem.SDMolSupplier(lig_input_file)
        mol = mols_suppl[0]
        mol3d = mol.GetConformer()
        # addtion用于增加微调范围
        size = get_size(mol, mol3d, addition=5)
        center = get_center(mol,mol3d)

        output_file = os.path.join(output_path,name)
        if not os.path.exists(output_file):
            os.mkdir(output_file)
            
        lig_pdbqt = os.path.join(output_file, f'{name}_ligand_pred.pdbqt')
        rec_pdbqt = os.path.join(output_file, f'{rec_name}.pdbqt')
        #将文件格式转换为pdbqt，以用于AutoDock-vina的输入
        subprocess.run(f'obabel {lig_input_file} -O {lig_pdbqt}', shell=True)
        subprocess.run(f'obabel {rec_input_file} -xr -O {rec_pdbqt}', shell=True)

        v = Vina(sf_name='vina')
        v.set_receptor(f'{rec_pdbqt}')
        v.set_ligand_from_file(f'{lig_pdbqt}')
        v.compute_vina_maps(center=center,box_size=size)
        v.dock(exhaustiveness=20,n_poses=5)
        v.write_poses(f'{output_file}/{name}_vina_out.pdbqt', overwrite=True)
        print(f"{name} is done!")

        #subprocess.run(f"./Vina/vina --receptor {rec_pdbqt} --ligand {lig_pdbqt} --center_x {center_x} --center_y {center_y} --center_z {center_z}\
        #--size_x {size_x} --size_y {size_y} --size_z {size_z} --num_modes 5 --energy_range 4 --exhaustiveness 20")
        