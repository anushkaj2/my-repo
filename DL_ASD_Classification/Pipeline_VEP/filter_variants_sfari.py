import pandas as pd
import os
import re
import glob
import multiprocessing
import time
import subprocess
from functools import partial


# This code filters VCF files for genes of interest and then annotates them with VEP


class vepFilter:
    def __init__(self, sfarigeneselected, genelistpython, input_dir, output_dir, annotated_dir, vep_path=None):
        self.sfarigeneselected = sfarigeneselected
        self.genelistpython = genelistpython
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.annotated_dir = annotated_dir
        self.genes = set()
        # VEP command path
        self.vep_path = vep_path if vep_path else "vep"

    def generate_gene_list(self):
        genes_df = pd.read_csv(self.sfarigeneselected)
        output_txt_file = self.genelistpython
        formatted_genes = ",".join(f'"{gene}"' for gene in genes_df.iloc[:, 0])
        with open(output_txt_file, "w") as f:
            f.write(formatted_genes)
        return self.genelistpython

    def load_gene_list(self):
        with open(self.genelistpython, "r") as f:
            gene_list_text = f.read().strip()
            self.genes = set(re.findall(r'"([^"]+)"', gene_list_text))
        print(f"Loaded: {len(self.genes)} number of genes for filtering .gvcf files")
        return self.genes

    def process_vcf(self, vcf_file):
        basename = os.path.basename(vcf_file)
        output_file = os.path.join(self.output_dir, f"{os.path.splitext(basename)[0]}_filtered.vcf")
        start_time = time.time()
        count_total = 0
        count_kept = 0

        with open(vcf_file, "r") as infile, open(output_file, 'w') as outfile:
            for line in infile:
                if line.startswith("#"):
                    outfile.write(line)
                    continue
                count_total += 1
                # Check if any gene from our list is in the annotation field
                keep = any(f"|{gene}|" in line for gene in self.genes)
                if keep:
                    outfile.write(line)
                    count_kept += 1
        processing_time = time.time() - start_time
        return (output_file, count_total, count_kept, processing_time)

    def annotate_vcf(self, filtered_vcf_file):
        basename = os.path.basename(filtered_vcf_file)
        sample_id = basename.split("_filtered")[0]
        output_file = os.path.join(self.annotated_dir, f"{sample_id}_vep_filt.vcf")

        print(f"Annotating {basename}...")
        start_time = time.time()

        # VEP command with specific dbNSFP fields
        vep_cmd = [
            self.vep_path,
            "-i", filtered_vcf_file,
            "-o", output_file,
            "--plugin",
            "dbNSFP,/home/ibab/sem4/datasets/databases/dbnsfp/dbNSFP5.0a/dbNSFP5.0a_grch38.gz,gnomAD4.1_joint_AF,gnomAD4.1_joint_AFR_AF,gnomAD4.1_joint_AMR_AF,gnomAD4.1_joint_ASJ_AF,gnomAD4.1_joint_EAS_AF,gnomAD4.1_joint_FIN_AF,gnomAD4.1_joint_NFE_AF,gnomAD4.1_joint_SAS_AF,gnomAD4.1_joint_MID_AF,gnomAD4.1_joint_AMI_AF,bStatistic,GERP++_RS,phastCons100way_vertebrate,phastCons470way_mammalian,phastCons17way_primate,phyloP100way_vertebrate,phyloP470way_mammalian,phyloP17way_primate,PROVEAN_score,CADD_phred,MutationAssessor_score,SIFT_score,SIFT4G_score,aapos,APPRIS,codon_degeneracy,codonpos,Interpro_domain,Reliability_index",
            "--tab",
            "--fork", "4",
            "--cache",
            "--assembly", "GRCh38",
            "--offline"
        ]

        try:
            print(f"Running VEP on {basename}...")
            vep_command = " ".join([str(arg) for arg in vep_cmd])
            process = subprocess.run(vep_command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

            if process.returncode != 0:
                print(f"Error annotating {basename}:\n{process.stderr}")
                return (filtered_vcf_file, output_file, time.time() - start_time, False)

            processing_time = time.time() - start_time
            return (filtered_vcf_file, output_file, processing_time, True)

        except Exception as e:
            print(f"Exception while annotating {basename}: {str(e)}")
            return (filtered_vcf_file, output_file, time.time() - start_time, False)

    def process_and_annotate_vcf(self, vcf_file):
        filtered_file, count_total, count_kept, filter_time = self.process_vcf(vcf_file)

        if count_kept > 0:
            _, annotated_file, annotation_time, success = self.annotate_vcf(filtered_file)
            return (vcf_file, filtered_file, annotated_file, count_total, count_kept,
                    filter_time, annotation_time, success)
        else:
            return (vcf_file, filtered_file, None, count_total, 0, filter_time, 0, True)

    def process_all_vcfs(self):
        if not self.input_dir:
            raise ValueError("Input directory not specified. Please set input_dir before calling this method.")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)

        # Make sure genes are loaded
        if not self.genes:
            self.load_gene_list()

        # Get all VCF files
        vcf_files = glob.glob(f"{self.input_dir}/*.vcf") + glob.glob(f"{self.input_dir}/*.gvcf")
        print(f"Found {len(vcf_files)} VCF files to process")
        if not vcf_files:
            print(f"No VCF files found in {self.input_dir}")
            return []

        # Process files in parallel
        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing")

        # First just filter all files (this is safer for parallel processing)
        filter_func = partial(self.process_vcf)
        with multiprocessing.Pool(processes=num_cores) as pool:
            filtered_results = pool.map(filter_func, vcf_files)

        # Then annotate the filtered files that have variants
        filtered_files_with_variants = []
        filter_stats = {}
        for orig_vcf_file, filtered_result in zip(vcf_files, filtered_results):
            filtered_file, count_total, count_kept, filter_time = filtered_result
            basename = os.path.basename(orig_vcf_file)
            print(f"{basename}: {count_kept}/{count_total} variants kept ({filter_time:.2f}s filtering)")
            filter_stats[filtered_file] = (count_total, count_kept, filter_time)
            if count_kept > 0:
                filtered_files_with_variants.append(filtered_file)

        # Only annotate files that have variants
        annotation_results = []
        if filtered_files_with_variants:
            print(f"\nAnnotating {len(filtered_files_with_variants)} filtered files with variants...")

            # Use parallel annotation to speed up processing
            # This is faster if your machine has enough resources
            # and your VEP is already properly configured in your environment
            annotate_func = partial(self.annotate_vcf)
            with multiprocessing.Pool(processes=min(num_cores, 4)) as pool:  # Limit to 4 processes to avoid overloading
                annotation_results = pool.map(annotate_func, filtered_files_with_variants)

        # Compile final results
        final_results = []
        for vcf_file in vcf_files:
            basename = os.path.basename(vcf_file)
            filtered_file = os.path.join(self.output_dir, f"{os.path.splitext(basename)[0]}_filtered.vcf")
            count_total, count_kept, filter_time = filter_stats.get(filtered_file, (0, 0, 0))
            # Find matching annotation result
            annotated_file = None
            annotation_time = 0
            success = True
            for f_file, a_file, a_time, a_success in annotation_results:
                if f_file == filtered_file:
                    annotated_file = a_file
                    annotation_time = a_time
                    success = a_success
                    break
            final_results.append((vcf_file, filtered_file, annotated_file, count_total, count_kept,
                                  filter_time, annotation_time, success))

        # Report results
        total_variants = 0
        total_kept = 0
        total_filter_time = 0
        total_annotation_time = 0
        successful_annotations = 0

        print("\n----- Final Results -----")
        for vcf_file, filtered_file, annotated_file, count_total, count_kept, filter_time, annotation_time, success in final_results:
            total_variants += count_total
            total_kept += count_kept
            total_filter_time += filter_time
            total_annotation_time += annotation_time
            basename = os.path.basename(vcf_file)
            print(f"{basename}: {count_kept}/{count_total} variants kept ({filter_time:.2f}s filtering)")
            if annotated_file:
                if success:
                    successful_annotations += 1
                    print(f"  → Annotated to {os.path.basename(annotated_file)} ({annotation_time:.2f}s annotation)")
                else:
                    print(f"  → Annotation failed")
            elif count_kept > 0:
                print(f"  → Not annotated")
            else:
                print(f"  → No variants to annotate")

        print(f"\nAll files processed:")
        print(f"- Total: {total_kept}/{total_variants} variants kept")
        print(f"- Filter time: {total_filter_time:.2f} seconds")
        print(f"- Annotation time: {total_annotation_time:.2f} seconds")
        print(f"- Successfully annotated: {successful_annotations} files")
        print(f"- Filtered results: {self.output_dir}")
        print(f"- Annotated results: {self.annotated_dir}")

        return final_results

    def run_pipeline(self, generate_new_gene_list=False):
        if generate_new_gene_list:
            self.generate_gene_list()
        self.load_gene_list()
        return self.process_all_vcfs()


def main():
    sfari_gene_file = "/home/ibab/sem4/created/sfari_gene_selected.csv"
    gene_list_file = "/home/ibab/sem4/created/gene_list_python.txt"
    input_vcf_dir = "/home/ibab/filtered"
    filtered_dir = "/home/ibab/filtered_vcfs"
    annotated_dir = "/home/ibab/annotated_vep_dir"
    vep_path = "/home/ibab/anaconda3/envs/vep/bin/vep"

    vep_pipeline = vepFilter(sfari_gene_file, gene_list_file, input_vcf_dir, filtered_dir, annotated_dir, vep_path)
    vep_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
