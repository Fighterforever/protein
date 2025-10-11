import os
from random import sample
import torch
from torch import nn
from Bio.PDB import PDBParser, FastMMCIFParser, PDBIO, StructureBuilder
from pe.common import residue_constants as rc
from pe.common import convert, util
from pe.model import modules, preprocess
from pe.data import loader, make_database
from statistics import mean
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Get LE embedding from pdb file')
    parse.add_argument('input_folder', type=str, help='Where pdf files are saved')
    parse.add_argument('output_folder', type=str, help='Where output files are saved')
    parse.add_argument('output_dim', type=int, default=21, help='output dimention')
    args = parse.parse_args()
    return args

def get_local_environment(pdb_file):
    device = torch.device('cpu')
    # 读取pdb文件
    protein_parser = PDBParser(QUIET=True)
    structure = protein_parser.get_structure('none', pdb_file)
    models = list(structure.get_models())
    model = models[0]

    original_residues = []
    chains = list(model.get_chains())

    # 得到原始序列
    for chain in chains:
        for res in chain.get_residues():
            original_residues.append(rc.restype_3to1.get(res.resname, 'X'))
    original_residues = ''.join(original_residues)

    # 保存只含主要原子的坐标
    io = PDBIO()
    io.set_structure(structure)
    main_chain_dir = "Temp/temp_" + pdb_file.split("/")[-1]
    io.save(main_chain_dir, convert.CustomSelect(only_main=True))

    seqs = []
    accs = []
    len_stage_1 = int(len(original_residues) / 1)
    len_stage_2 = int(len(original_residues) / 5)
    ks = [5] * len_stage_1 + [1] * len_stage_2

    num_class = len(rc.restypes_with_x)

    # 加载预训练模型
    body = modules.Transformer(46, num_class, 256, nhead=16, nlayer=3,device=device)
    body.load_state_dict(torch.load("best.pkl", map_location=device)['model_state_dict'])

    body.eval()

    validate = loader.get_loader(
        make_database.make_dataset(main_chain_dir, radius=3.5),
        device=util.get_model_device(body),
        batch_size=1)

    total_loss = 0
    losses = []
    is_optimal = []
    total_loss_by_chain = {}
    total_acc = 0
    n_sample = 0
    process = preprocess.PreProcess()
    loss = nn.CrossEntropyLoss(reduction='none')

    result = {}
    logits, features = [], []
    for aa in validate:
        processed = process(aa['feature'])
        log, feat = body(processed, mask=aa['mask'])
        logits.append(log)
        features.append(feat)
    result['logits'] = torch.cat(logits)
    result['features'] = torch.cat(features)
    return result


if __name__ == '__main__':
    args = parse_args()
    pdb_path = args.input_folder
    output_folder = args.output_folder
    dim = args.output_dim

    pdb = os.listdir(pdb_path)
    for i in tqdm(pdb):
        output = output_folder + i.replace("pdb","txt")
        if os.path.exists(output):
            continue
        else:
            # print(i)
            if dim == 21:
                dim21 = get_local_environment(pdb_file=pdb_path+i)["features"]
                dim21 = dim21.detach().numpy()
                np.savetxt(output,dim21)
            if dim == 100:
                dim100 = get_local_environment(pdb_file=pdb_path+i)["logits"]
                dim100 = dim100.detach().numpy()
                np.savetxt(output,dim100)
