from pathlib import Path
from Bio import SeqIO
from gpcrscan.data.objects import SeqDict
from gpcrscan.data.objects import SeqInfo


def parse_fasta(fasta: str | Path, splitter: str = "|"):
    """Parse fasta file."""
    SEQ_DICT = {}
    duplicates = []

    for record in SeqIO.parse(fasta, "fasta"):
        id = record.description.split()[-1]
        id = id.upper()
        uniprot_id = id.split("|")[0]
        if uniprot_id in SEQ_DICT:
            print("Duplicated uniprot_id: ", uniprot_id)
            duplicates.append(uniprot_id)
            continue

        SEQ_DICT[uniprot_id] = SeqInfo(id=id, description=record.description, seq=record.seq, splitter=splitter)
    print("Number of duplicated uniprot_ids: ", len(duplicates))
    # return SEQ_DICT
    return SeqDict(SEQ_DICT)  # type: ignore


# dataclass that holds the dict of SeqInfo objects
