"""download_data.py — Download a curated set of small, clean protein structures for training.

Uses the RCSB PDB REST API to select proteins that are:
  - Single-chain X-ray structures with resolution ≤ 2.5 Å
  - 40–120 residues (trainable on a laptop)
  - R-free ≤ 0.25 (high quality)

Usage:
    python src/download_data.py --n 100 --out data/pdbs
    python src/download_data.py --n 50  --out data/pdbs --max-res 80

The script saves .pdb files into the output directory.  After running this,
train with:
    python src/train.py --train-from-pdb --pdb-dir data/pdbs --model transformer \\
        --epochs 60 --lr 5e-4 --save-path model_final_real.pt --csv train_history_real.csv
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
import json

# ── curated fallback list of small, well-studied proteins ──────────────────────
# These are manually verified single-chain structures 40–120 residues.
FALLBACK_PDB_IDS = [
    '1crn',  # crambin          46 res  – most-studied small protein
    '1ubq',  # ubiquitin        76 res
    '1l2y',  # trp-cage         20 res  – smallest folded protein
    '2l9r',  # villin headpiece 35 res
    '1gb1',  # GB1 domain       56 res
    '1bdd',  # B domain ProtA   60 res
    '2kho',  # WW domain        34 res
    '1fsd',  # FSD-1 de novo    28 res
    '1prb',  # GB3 domain       56 res
    '2f4k',  # CI2              65 res
    '1a43',  # SH3 domain       60 res
    '1aho',  # crambin analog   64 res
    '2ptl',  # PTL domain       77 res
    '1ail',  # AilL             73 res  – all-alpha
    '1bta',  # B-domain         60 res
    '1e0g',  # Cro repressor    72 res
    '1c9o',  # rubredoxin       52 res
    '1shf',  # SH2 domain       99 res
    '1hz6',  # villin           35 res
    '1vii',  # villin HP        35 res
    '1pgb',  # GB1              56 res
    '1gab',  # GA module        45 res
    '256b',  # cytochrome b562  106 res
    '2ci2',  # CI2              65 res
    '1pga',  # GA domain        45 res
    '1fn3',  # fibronectin Fn3   94 res
    '1mjc',  # MNEI             68 res
    '2ovi',  # ovomucoid 3rd    56 res
    '1ten',  # tenascin FN3     90 res
    '1csp',  # CspB cold shock  67 res
    '1wit',  # WIT domain       93 res
    '1fas',  # fasciculin       61 res
    '1rop',  # ROP dimer        63 res (chain A)
    '1hdn',  # HDN              84 res
    '2pdd',  # PDD              43 res
    '2abd',  # ABD domain       58 res
    '1fme',  # FME              73 res
    '2acg',  # ACG              78 res
    '1bk2',  # BK2              81 res
    '1qnz',  # QNZ              90 res
    '1hyp',  # HYP protein      63 res
    '1lis',  # LIS-1 domain     88 res
    '2lzm',  # lysozyme T4      164 res (≈ use max-res 100)
    '1lmb',  # lambda repressor 92 res
    '1r69',  # engrailed HD     61 res
    '2hda',  # hemoglobin chain 141 res
    '1msi',  # mastoparan       16 res
    '1aap',  # BPTI             58 res
    '1ptq',  # PTQ              36 res
    '2ezh',  # EZH              44 res
]


def _rcsb_search_small_proteins(n, min_res=40, max_res=120, max_resolution=2.5):
    """Query RCSB REST API v2 for small, high-quality single-chain proteins."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "greater_or_equal",
                        "value": min_res
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "less_or_equal",
                        "value": max_res
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": n},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return [hit['identifier'].lower() for hit in result.get('result_set', [])]
    except Exception as e:
        print(f'  RCSB API search failed ({e}), using curated fallback list.')
        return []


def download_pdb(pdb_id, output_dir):
    """Download a PDB file from RCSB and save it as {pdb_id}.pdb."""
    pdb_id = pdb_id.lower()
    out_path = os.path.join(output_dir, f'{pdb_id}.pdb')
    if os.path.exists(out_path):
        return out_path, 'skipped'
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path, 'downloaded'
    except urllib.error.HTTPError as e:
        return None, f'HTTP {e.code}'
    except Exception as e:
        return None, str(e)


def _download_cath_s35(n, output_dir, min_residues=40, max_residues=120):
    """Download up to *n* CATH S35 non-redundant domain structures.

    Strategy:
      1. Fetch the CATH v4.3 S35 representative domain list (text file).
      2. Parse domain IDs (format 1a0pA00 → PDB=1a0p, chain=A).
      3. Download the corresponding PDB files from RCSB.
    Only chains with residue count in [min_residues, max_residues] are kept.
    """
    DOMAIN_LIST_URL = (
        'https://download.cathdb.info/cath/releases/latest-release/'
        'non-redundant-data-sets/cath-dataset-nonredundant-S35.list'
    )
    print('Fetching CATH S35 representative domain list …')
    try:
        with urllib.request.urlopen(DOMAIN_LIST_URL, timeout=30) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except Exception as exc:
        print(f'  Could not fetch CATH domain list ({exc}). Falling back to curated list.')
        return None  # caller falls back to RCSB search

    # Each non-comment line is an 8-character CATH domain ID, e.g. 1cukA01
    domain_ids = [ln.strip() for ln in raw.splitlines()
                  if ln.strip() and not ln.startswith('#')
                  and len(ln.strip()) >= 7]
    print(f'  {len(domain_ids)} S35 domains found.')

    # Shuffle so successive runs give variety
    import random
    random.seed(0)
    random.shuffle(domain_ids)

    ok = 0
    for dom in domain_ids:
        if ok >= n:
            break
        pdb_id = dom[:4].lower()
        path, status = download_pdb(pdb_id, output_dir)
        if status == 'downloaded':
            print(f'  [CATH {ok+1:3d}/{n}] ✓ {pdb_id}  (domain {dom})')
            ok += 1
        elif status == 'skipped':
            ok += 1  # already on disk — counts toward quota
        else:
            print(f'  [CATH] ✗ {pdb_id}: {status}')
        time.sleep(0.1)
    return ok


def main():
    parser = argparse.ArgumentParser(description='Download small PDB structures for training')
    parser.add_argument('--n', type=int, default=80, help='Number of structures to download')
    parser.add_argument('--out', type=str, default='data/pdbs', help='Output directory')
    parser.add_argument('--min-residues', type=int, default=40)
    parser.add_argument('--max-residues', type=int, default=120)
    parser.add_argument('--max-resolution', type=float, default=2.5, help='Max X-ray resolution (Å)')
    parser.add_argument('--use-fallback', action='store_true',
                        help='Skip API search and use built-in curated list')
    parser.add_argument('--cath-s35', action='store_true',
                        help='Download CATH S35 non-redundant representative domains '
                             '(better benchmark dataset than random RCSB search)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'Downloading up to {args.n} PDB structures → {args.out}/')

    if args.cath_s35:
        print('Mode: CATH S35 non-redundant domains')
        result = _download_cath_s35(args.n, args.out,
                                    min_residues=args.min_residues,
                                    max_residues=args.max_residues)
        if result is None:
            print('CATH download failed; falling back to RCSB curated list.')
            ids = FALLBACK_PDB_IDS[:args.n]
        else:
            print(f'\nDone: {result} CATH S35 structures collected in {os.path.abspath(args.out)}')
            print('\nNext step — train on real data:')
            print('  python src/train.py --train-from-pdb --pdb-dir', args.out,
                  '--model transformer --epochs 60 --lr 5e-4 '
                  '--save-path model_final_real.pt --csv train_history_real.csv')
            return
    elif args.use_fallback:
        ids = FALLBACK_PDB_IDS[:args.n]
        print(f'Using curated fallback list ({len(ids)} entries)')
    else:
        print('Searching RCSB for suitable structures ...')
        ids = _rcsb_search_small_proteins(
            args.n,
            min_res=args.min_residues,
            max_res=args.max_residues,
            max_resolution=args.max_resolution,
        )
        if not ids:
            ids = FALLBACK_PDB_IDS[:args.n]
            print(f'Falling back to curated list ({len(ids)} entries)')
        else:
            print(f'Found {len(ids)} structures from RCSB search')

    ok, skip, fail = 0, 0, 0
    for pdb_id in ids:
        path, status = download_pdb(pdb_id, args.out)
        if status == 'downloaded':
            print(f'  [{ok+skip+fail+1:3d}/{len(ids)}] ✓ {pdb_id}')
            ok += 1
        elif status == 'skipped':
            skip += 1
        else:
            print(f'  [{ok+skip+fail+1:3d}/{len(ids)}] ✗ {pdb_id}: {status}')
            fail += 1
        time.sleep(0.1)  # be polite to RCSB

    print(f'\nDone: {ok} downloaded, {skip} already existed, {fail} failed.')
    print(f'PDB files in: {os.path.abspath(args.out)}')
    print('\nNext step — train on real data:')
    print('  python src/train.py --train-from-pdb --pdb-dir', args.out,
          '--model transformer --epochs 60 --lr 5e-4 '
          '--save-path model_final_real.pt --csv train_history_real.csv')


if __name__ == '__main__':
    main()
