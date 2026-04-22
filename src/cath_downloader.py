#!/usr/bin/env python3
"""cath_downloader.py — Download CATH S35 non-redundant protein domains at scale.

CATH S35 is the gold-standard non-redundant dataset used in published protein
structure prediction papers. Proteins are clustered at 35% sequence identity,
giving diverse coverage of structural space without redundancy bias.

Usage:
    python src/cath_downloader.py --n 800 --out data/cath_s35 --min-res 40 --max-res 120
    python src/cath_downloader.py --n 200 --out data/cath_s35 --min-res 40 --max-res 60
"""
import argparse
import os
import sys
import time
import urllib.request
import urllib.error

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils

CATH_S35_URL = (
    'https://download.cathdb.info/cath/releases/latest-release/'
    'non-redundant-data-sets/cath-dataset-nonredundant-S35.list'
)

RCSB_PDB_URL = 'https://files.rcsb.org/download/{}.pdb'


def fetch_cath_domain_list(cache_path='data/cath_s35_domains.txt'):
    """Fetch the CATH S35 domain list and cache it locally."""
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)

    if os.path.exists(cache_path):
        print(f'Using cached CATH domain list: {cache_path}')
        with open(cache_path) as f:
            lines = f.readlines()
    else:
        print(f'Fetching CATH S35 domain list from {CATH_S35_URL} ...')
        try:
            with urllib.request.urlopen(CATH_S35_URL, timeout=30) as r:
                content = r.read().decode('utf-8')
            with open(cache_path, 'w') as f:
                f.write(content)
            lines = content.splitlines(keepends=True)
            print(f'  Fetched {len(lines)} lines, cached to {cache_path}')
        except Exception as e:
            print(f'  ERROR fetching CATH list: {e}')
            return []

    # CATH domain IDs look like: 1cukA01  (pdb=1cuk, chain=A, domain=01)
    domains = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and len(line) >= 7:
            domains.append(line.split()[0])
    print(f'  {len(domains)} CATH S35 domains available')
    return domains


def domain_to_pdb_chain(domain_id):
    """Parse CATH domain ID → (pdb_id, chain).
    Example: '1cukA01'  →  ('1cuk', 'A')
    """
    pdb_id = domain_id[:4].lower()
    chain  = domain_id[4].upper()
    return pdb_id, chain


def download_and_validate(pdb_id, chain, output_dir, min_res=40, max_res=120):
    """Download PDB, extract chain, validate residue count.
    Returns (seq, coords, actual_length) or raises ValueError.
    """
    pdb_path = utils.fetch_pdb(pdb_id, output_dir=output_dir)
    seq    = utils.pdb_sequence(pdb_path, chain=chain, max_residues=max_res + 10)
    coords = utils.pdb_ca_coords(pdb_path, chain=chain, max_residues=max_res + 10)
    N = min(len(seq), len(coords))
    if N < min_res:
        raise ValueError(f'Too short: {N} < {min_res}')
    if N > max_res + 20:   # allow some slack; we'll crop during training
        N = max_res
    return seq[:N], coords[:N], N


def download_cath_s35(n, output_dir='data/cath_s35', min_res=40, max_res=120,
                      shuffle_seed=42, cache_path='data/cath_s35_domains.txt'):
    """Main entry point: download n validated CATH S35 domains.

    Returns list of (pdb_id, chain, n_residues) for successfully downloaded proteins.
    """
    import random
    os.makedirs(output_dir, exist_ok=True)

    domains = fetch_cath_domain_list(cache_path=cache_path)
    if not domains:
        print('No domains fetched — aborting.')
        return []

    rng = random.Random(shuffle_seed)
    rng.shuffle(domains)

    ok, skipped = [], 0
    manifest_path = os.path.join(output_dir, 'manifest.tsv')

    # Load already-downloaded proteins from manifest to allow resume
    already = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    already[parts[0]] = (parts[1], int(parts[2]))
    if already:
        print(f'  Resuming: {len(already)} proteins already in manifest')
        ok = [(pid, chain, length) for pid, (chain, length) in already.items()]

    print(f'\nDownloading up to {n} CATH S35 proteins ({min_res}–{max_res} residues) ...\n')
    for dom in domains:
        if len(ok) >= n:
            break
        pdb_id, chain = domain_to_pdb_chain(dom)
        key = f'{pdb_id}_{chain}'
        if key in already:
            continue
        try:
            seq, coords, length = download_and_validate(
                pdb_id, chain, output_dir, min_res=min_res, max_res=max_res)
            ok.append((pdb_id, chain, length))
            already[key] = (chain, length)
            # Append to manifest
            with open(manifest_path, 'a') as f:
                f.write(f'{key}\t{chain}\t{length}\n')
            print(f'  [{len(ok):4d}/{n}] {pdb_id} chain {chain}: {length} aa')
        except Exception as e:
            skipped += 1
            if skipped % 50 == 0:
                print(f'  ... skipped {skipped} so far ({e})')

    print(f'\nDone: {len(ok)} proteins downloaded, {skipped} skipped')
    print(f'Manifest: {manifest_path}')
    return ok


def main():
    p = argparse.ArgumentParser(description='Download CATH S35 non-redundant protein domains')
    p.add_argument('--n',       type=int,   default=800,          help='Target number of proteins')
    p.add_argument('--out',     type=str,   default='data/cath_s35', help='Output PDB directory')
    p.add_argument('--min-res', type=int,   default=40,           help='Minimum residue count')
    p.add_argument('--max-res', type=int,   default=120,          help='Maximum residue count')
    p.add_argument('--seed',    type=int,   default=42,           help='Shuffle seed')
    p.add_argument('--cache',   type=str,   default='data/cath_s35_domains.txt')
    args = p.parse_args()

    proteins = download_cath_s35(
        n=args.n, output_dir=args.out,
        min_res=args.min_res, max_res=args.max_res,
        shuffle_seed=args.seed, cache_path=args.cache,
    )
    if proteins:
        lengths = [l for _, _, l in proteins]
        import numpy as np
        print(f'\nLength stats: mean={np.mean(lengths):.1f}  '
              f'min={min(lengths)}  max={max(lengths)}  '
              f'median={int(np.median(lengths))}')


if __name__ == '__main__':
    main()
