import os
from model.init import seed_everything
from dataset.classification import *
from dataset.extraction import *
from dataset.generation import *
from dataset.config import get_models_evaluate
from metric.extraction import print_metrics_ext
from metric.generation import print_metrics_gen
from metric.classification import print_metrics_clf


class EmptyArgs:
    def __init__(self):
        pass


args = EmptyArgs()

num_seed = 42
seed_everything(seed=num_seed)

# Configuration
num_bootstrap = 1000
path_dir_performance = "performance"
list_prompt_mode = ["direct", "cot", "direct-5-shot"]
model_to_evaluate = list_model = [
    "Baichuan-M1-14B-Instruct",
    "DeepSeek-R1",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Llama-70B",
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-32B",
    "gemma-2-9b-it",
    "gemma-2-27b-it",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.3-70B-Instruct",
    "Llama-4-Scout-17B-16E-Instruct",
    "Llama-3.1-Nemotron-70B-Instruct-HF",
    "meditron-7b",
    "meditron-70b",
    "MeLLaMA-13B-chat",
    "MeLLaMA-70B-chat",
    "Llama3-OpenBioLLM-8B",
    "Llama3-OpenBioLLM-70B",
    "MMed-Llama-3-8B",
    "Llama-3.1-8B-UltraMedical",
    "Llama-3-70B-UltraMedical",
    "Ministral-8B-Instruct-2410",
    "Mistral-Small-Instruct-2409",
    "Mistral-Small-24B-Instruct-2501",
    "Mistral-Small-3.1-24B-Instruct-2503",
    "Mistral-Large-Instruct-2411",
    "BioMistral-7B",
    "Phi-3.5-mini-instruct",
    "Phi-3.5-MoE-instruct",
    "Phi-4",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
    "QwQ-32B-Preview",
    "QWQ-32B",
    "Athene-V2-Chat",
    "Yi-1.5-9B-Chat-16K",
    "Yi-1.5-34B-Chat-16K",
    "gpt-35-turbo",
    "gpt-4o",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]

model_to_print = get_models_evaluate()

# Classfication
all_tasks_clf = {
    "ADE-Identification": Task_clf_ADE_ADE_identification,
    "BrainMRI-AIS": Task_clf_Brain_MRI_AIS,
    "Brateca-Hospitalization": Task_clf_Brateca_hospitalization,
    "Brateca-Mortality": Task_clf_Brateca_mortality,
    "Cantemist-Coding": Task_nor_Cantemist_CODING,
    "CARES-Area": Task_clf_CARES_area,
    "CARES-ICD10 Chapter": Task_clf_CARES_icd10_chapter,
    "CARES ICD10 Block": Task_nor_CARES_icd10_block,
    "CARES-ICD10 Subblock": Task_nor_CARES_icd10_sub_block,
    "C-EMRS": Task_clf_C_EMRS,
    "ClinicalNotes-UPMC": Task_clf_Clinical_Notes_UPMC,
    "PPTS": Task_clf_clinical_records_from_the_Mexican_Social_Security_Institute,
    "CLIP": Task_clf_CLIP,
    "DialMed": Task_clf_Dial_Med,
    "EHRQA-Primary department": Task_clf_EHRQA_primary_department,
    "EHRQA-Sub department": Task_clf_EHRQA_sub_department,
    "GOUT-CC-Consensus": Task_clf_GOUT_CC_consensus,
    "JP-STS": Task_clf_Japanese_Case_Reports,
    "MEDIQA 2019-RQE": Task_clf_MEDIQA_2019_Task2_RQE,
    "MedNLI": Task_clf_MedNLI,
    "MedSTS": Task_clf_MedSTS,
    "MTS": Task_clf_mtsamples,
    "MEDIQA 2023-sum-A": Task_clf_MTS_Dialog_MEDIQA_2023_sum_task_A,
    "RuMedDaNet": Task_clf_RuMedDaNet,
    "CBLUE-CDN": Task_nor_CHIP_CDN,
    "CHIP-CTC": Task_clf_CHIP_CTC,
    "IMCS-V2-DAC": Task_clf_IMCS_V2_DAC,
    "RuMedNLI": Task_clf_RuMedNLI,
    "CLISTER": Task_clf_CLISTER,
    "IFMIR-Incident type": Task_clf_IFMIR_IncidentType,
    "MIMIC-IV CDM": Task_clf_mimic_iv_CDM,
    "MIMIC-III Outcome.LoS": Task_clf_mimic_iii_outcome_LoS,
    "MIMIC-III Outcome.Mortality": Task_clf_mimic_iii_outcome_Mortality,
    "MIMIC-IV DiReCT.Dis": Task_clf_mimic_iv_DiReCT_Dis,
    "MIMIC-IV DiReCT.PDD": Task_clf_mimic_iv_DiReCT_PDD,
}

# Extraction
all_tasks_ext = {
    "ADE-Extraction": Task_ext_ADE_ADE_relation,
    "ADE-Drug dosage": Task_ext_ADE_Drug_dosage,
    "BARR2": Task_ext_BARR2_resolution,
    "Cantemis-NER": Task_ext_Cantemist_NER,
    "Cantemis-Norm": Task_ext_Cantemist_Norm,
    "CHIP-CDEE": Task_ext_CHIP_CDEE,
    "CodiEsp-ICD-10-CM": Task_ext_CLEF_ICD_10_CM,
    "CodiEsp-ICD-10-PCS": Task_ext_CLEF_ICD_10_PCS,
    "CLINpt-NER": Task_ext_CLINpt,
    "DiSMed-NER": Task_ext_DiSMed,
    "MIE": Task_ext_MIE,
    "Ex4CDS": Task_ext_Ex4CDS,
    "n2c2 2006-De-identification": Task_ext_n2c2_2006_De_Identification,
    "Medication extraction": Task_ext_i2b2_2009_Medication_Extraction_Challenge,
    "n2c2 2010-Concept": Task_ext_i2b2_2010_Relations_Challenge_concept,
    "n2c2 2010-Assertion": Task_ext_i2b2_2010_Relations_Challenge_assertion,
    "n2c2 2010-Relation": Task_ext_i2b2_2010_Relations_Challenge_relation,
    "n2c2 2014-De-identification": Task_ext_n2c2_2014_De_identification,
    "IMCS-V2-NER": Task_ext_IMCS_V2_NER,
    "meddocan": Task_ext_meddocan,
    "MTS-Temporal": Task_ext_MTSamples_temporal_annotation,
    "n2c2 2018-ADE&medication": Task_ext_n2c2_2018_Track2_Adverse_Drug_Events_and_Medication_Extraction,
    "NorSynthClinical-NER": Task_ext_NorSynthClinical_entity,
    "NorSynthClinical-RE": Task_ext_NorSynthClinical_relation,
    "NUBES": Task_ext_NUBES,
    "CHIP-MDCFNPC": Task_ext_CHIP_MDCFNPC,
    "IMCS-V2-SR": Task_ext_IMCS_V2_SR,
    "n2c2 2014-Diabetes": Task_ext_n2c2_2014_Heart_Disease_Challenge_Diabete,
    "n2c2 2014-CAD": Task_ext_n2c2_2014_Heart_Disease_Challenge_CAD,
    "n2c2 2014-Hyperlipidemia": Task_ext_n2c2_2014_Heart_Disease_Challenge_Hyperlipidemia,
    "n2c2 2014-Hypertension": Task_ext_n2c2_2014_Heart_Disease_Challenge_Hypertension,
    "n2c2 2014-Medication": Task_ext_n2c2_2014_Heart_Disease_Challenge_Medication,
    "CAS-label": Task_ext_CAS_label,
    "RuDReC-NER": Task_ext_RuDReC,
    "NorSynthClinical-PHI": Task_ext_NorSynthClinical_PHI,
    "RuCCoN": Task_ext_RuCCoN_NER,
    "BRONCO150-NER&Status": Task_ext_BRONCO150_NER_status,
    "CARDIO-DE": Task_ext_CARDIO_DE,
    "GraSSCo PHI": Task_ext_GraSSCo_PHI,
    "IFMIR-NER": Task_ext_IFMIR_NER,
    "IFMIR-NER&factuality": Task_ext_IFMIR_NER_factuality,
    "iCorpus": Task_ext_iCorpus,
}

# Generation
all_tasks_gen = {
    "cMedQA": Task_gen_cMedQA,
    "EHRQA-QA": Task_gen_EHRQA_qa,
    "MEDIQA 2023-chat-A": Task_gen_MTS_Dialog_MEDIQA_2023_chat_task_A,
    "MEDIQA 2023-sum-B": Task_gen_MTS_Dialog_MEDIQA_2023_sum_task_B,
    "MedDG": Task_gen_MedDG,
    "IMCS-V2-MRG": Task_gen_IMCS_V2_MRG,
    "CAS-evidence": Task_gen_CAS_evidence,
    "icliniq-10k": Task_gen_icliniq,
    "HealthCareMagic-100k": Task_gen_HealthCareMagic,
    "MIMIC-IV BHC": Task_gen_mimic_iv_BHC,
}


def evaluate(task):
    dict_prompt_model_performance = {}
    for prompt_mode in list_prompt_mode:
        dict_model_performance = task.evaluate_by_model(
            prompt_mode=prompt_mode,
            model_name=model_to_evaluate,
            bootstrap=num_bootstrap,
        )
        path_file_performance = (
            f"{path_dir_performance}/{task.name}.{prompt_mode}.performance.json"
        )
        # If the file exists, load the existing file and update the performance. Otherwise, create a new file.
        if os.path.exists(path_file_performance):
            with open(path_file_performance, "r") as f:
                dict_model_performance_old = json.load(f)
            for model_name in dict_model_performance.keys():
                dict_model_performance_old[model_name] = dict_model_performance[
                    model_name
                ]
            dict_model_performance = dict_model_performance_old
            with open(path_file_performance, "w") as f:
                json.dump(dict_model_performance, f, indent=4)
        else:
            with open(path_file_performance, "w") as f:
                json.dump(dict_model_performance, f, indent=4)
        dict_prompt_model_performance[prompt_mode] = dict_model_performance
    return dict_prompt_model_performance


def print_performance(dict_prompt_model_performance, task_type):
    dict_mode_performance = {}
    for prompt_mode in list_prompt_mode:
        print("Prompt Mode:", prompt_mode)
        if task_type == "clf":
            str_metrics = print_metrics_clf(
                dict_prompt_model_performance[prompt_mode], flag_print_missing=False
            )
        elif task_type == "ext":
            str_metrics = print_metrics_ext(
                dict_prompt_model_performance[prompt_mode], flag_print_missing=False
            )
        elif task_type == "gen":
            str_metrics = print_metrics_gen(
                dict_prompt_model_performance[prompt_mode], flag_print_missing=False
            )
        else:
            raise ValueError("Invalid task type")
        print(str_metrics)
        print("===============================")
        dict_mode_performance[prompt_mode] = str_metrics
    return dict_mode_performance


def print_all_performance():
    # Classification
    print("Classification")
    for prompt_mode in list_prompt_mode:
        print(f"Prompt Mode: {prompt_mode}")
        for task_name in all_tasks_clf.keys():
            path_file_performance = (
                f"{path_dir_performance}/{task_name}.{prompt_mode}.performance.json"
            )
            with open(path_file_performance, "r") as f:
                dict_model_performance = json.load(f)
            str_metrics = print_metrics_clf(
                dict_model_performance, model_to_print, flag_print_missing=False
            )
            print(str_metrics)
        print("===============================")
    print()

    # Extraction
    print("Extraction")
    for prompt_mode in list_prompt_mode:
        print(f"Prompt Mode: {prompt_mode}")
        for task_name in all_tasks_ext.keys():
            path_file_performance = (
                f"{path_dir_performance}/{task_name}.{prompt_mode}.performance.json"
            )
            with open(path_file_performance, "r") as f:
                dict_model_performance = json.load(f)
            str_metrics = print_metrics_ext(
                dict_model_performance, model_to_print, flag_print_missing=False
            )
            print(str_metrics)
        print("===============================")
    print()

    # Generation
    print("Generation")
    for prompt_mode in list_prompt_mode:
        print(f"Prompt Mode: {prompt_mode}")
        for task_name in all_tasks_gen.keys():
            path_file_performance = (
                f"{path_dir_performance}/{task_name}.{prompt_mode}.performance.json"
            )
            with open(path_file_performance, "r") as f:
                dict_model_performance = json.load(f)
            str_metrics = print_metrics_gen(
                dict_model_performance, model_to_print, flag_print_missing=False
            )
            print(str_metrics)
        print("===============================")
    print()


def evaluate_all():
    for task, evaluation_function in all_tasks_clf.items():
        print(task)
        task = evaluation_function(args=args, task=task)
        dict_prompt_model_performance = evaluate(task)
        dict_mode_performance = print_performance(dict_prompt_model_performance, "clf")
        print()
    for task, evaluation_function in all_tasks_ext.items():
        print(task)
        task = evaluation_function(args=args, task=task)
        dict_prompt_model_performance = evaluate(task)
        dict_mode_performance = print_performance(dict_prompt_model_performance, "ext")
        print()
    for task, evaluation_function in all_tasks_gen.items():
        print(task)
        task = evaluation_function(args=args, task=task)
        dict_prompt_model_performance = evaluate(task)
        dict_mode_performance = print_performance(dict_prompt_model_performance, "gen")
        print()


if __name__ == "__main__":
    evaluate_all()
    print_all_performance()
