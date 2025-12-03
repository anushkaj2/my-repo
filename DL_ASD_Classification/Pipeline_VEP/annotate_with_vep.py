# Run this code on CPU as this utilises a lot of file operations
import os
import glob
import time
import subprocess
import multiprocessing

class VEPAnnotator:
    def __init__(self, input_dir, annotated_dir, vep_path="vep", fork=4):
        self.input_dir = input_dir
        self.annotated_dir = annotated_dir
        self.vep_path = vep_path
        self.fork = fork
        os.makedirs(self.annotated_dir, exist_ok=True)

    def annotate_vcf(self, vcf_file):
        basename = os.path.basename(vcf_file)
        sample_id = basename.replace("_filtered.vcf", "")
        output_file = os.path.join(self.annotated_dir, f"{sample_id}_vep_filt.vcf")
        print(f"Annotating {basename}...")
        start = time.time()
        vep_cmd = [
            self.vep_path,
            "-i", vcf_file,
            "-o", output_file,
            "--plugin", "dbNSFP,/home/ibab/sem4/datasets/databases/dbnsfp/dbNSFP5.0a/dbNSFP5.0a_grch38.gz,"
            "gnomAD4.1_joint_AF,gnomAD4.1_joint_AFR_AF,gnomAD4.1_joint_AMR_AF,gnomAD4.1_joint_ASJ_AF,"
            "gnomAD4.1_joint_EAS_AF,gnomAD4.1_joint_FIN_AF,gnomAD4.1_joint_NFE_AF,gnomAD4.1_joint_SAS_AF,"
            "gnomAD4.1_joint_MID_AF,gnomAD4.1_joint_AMI_AF,bStatistic,GERP++_RS,phastCons100way_vertebrate,"
            "phastCons470way_mammalian,phastCons17way_primate,phyloP100way_vertebrate,"
            "phyloP470way_mammalian,phyloP17way_primate,PROVEAN_score,CADD_phred,"
            "MutationAssessor_score,SIFT_score,SIFT4G_score,aapos,APPRIS,codon_degeneracy,"
            "codonpos,Interpro_domain,Reliability_index",
            "--tab",
            "--fork", str(self.fork),
            "--cache",
            "--assembly", "GRCh38",
            "--offline"
        ]
        cmd_str = " ".join(vep_cmd)
        result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"Error annotating {basename}: {result.stderr}")
            return False
        print(f"Finished {basename} in {elapsed:.2f}s")
        return True

    def run(self):
        vcfs = glob.glob(os.path.join(self.input_dir, "*_filtered.vcf"))
        if not vcfs:
            print("No filtered VCF files found to annotate.")
            return
        num_procs = min(multiprocessing.cpu_count(), self.fork)
        print(f"Annotating {len(vcfs)} files using {num_procs} processes...")
        with multiprocessing.Pool(processes=num_procs) as pool:
            pool.map(self.annotate_vcf, vcfs)

if __name__ == "__main__":
    input_dir = "/home/ibab/filtered"  # directory with *_filtered.vcf files
    annotated_dir = "/home/ibab/annotated_vep_dir"
    vep_path = "/home/ibab/anaconda3/envs/vep/bin/vep"

    annotator = VEPAnnotator(input_dir, annotated_dir, vep_path)
    annotator.run()
