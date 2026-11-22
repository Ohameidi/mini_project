from pathlib import Path
import time
import mne
from mne.datasets import eegbci


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "datasets"


def download_all_subjects(
    subjects,
    runs,
    path=None,
    retries=3,
    delay_seconds=2.0,
):
    target_dir = Path(path).expanduser() if path is not None else DEFAULT_DATASET_PATH
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = {}
    failed_subjects = []

    total = len(subjects)
    for idx, subj in enumerate(subjects, start=1):
        success = False
        for attempt in range(1, retries + 1):
            try:
                file_paths = eegbci.load_data([subj], runs, path=str(target_dir))
                downloaded_files[subj] = file_paths
                print(f"[{idx}/{total}] Subject {subj:03d} erfolgreich geladen ({len(file_paths)} Dateien).")
                success = True
                break
            except Exception as exc:
                print(
                    f"[{idx}/{total}] Versuch {attempt}/{retries} für Subject {subj:03d} "
                    f"fehlgeschlagen: {exc}"
                )
                if attempt < retries:
                    time.sleep(delay_seconds * attempt)

        if not success:
            failed_subjects.append(subj)

    return downloaded_files, failed_subjects


def load_subject_raw(subj, runs, path=None):
    target_dir = Path(path).expanduser() if path is not None else DEFAULT_DATASET_PATH
    file_paths = eegbci.load_data([subj], runs, path=str(target_dir))

    raw_list = []
    for fp in file_paths:
        raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")
        raw_list.append(raw)

    raw_subj = mne.concatenate_raws(raw_list)
    events, event_id = mne.events_from_annotations(raw_subj, verbose=False)
    return raw_subj, events, event_id


if __name__ == "__main__":
    all_subjects = list(range(1, 110))  # 1 bis 109
    runs = [6, 10, 14]
    dataset_path = DEFAULT_DATASET_PATH

    files_by_subj, failed = download_all_subjects(
        subjects=all_subjects,
        runs=runs,
        path=dataset_path,
    )

    total_files = sum(len(v) for v in files_by_subj.values())
    print("\n--- Summery ---")
    print(f"Subjects: {len(files_by_subj)} / {len(all_subjects)}")
    print(f"total files: {total_files}")
    print(f"failed Subjects: {failed}")