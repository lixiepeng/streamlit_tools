import json
import multiprocessing as mp
import os
import sys

from pyresparser import ResumeParser


def resume_result_wrapper(resume):
    parser = ResumeParser(resume)
    return parser.get_extracted_data()


if __name__ == '__main__':
    resume_dir = sys.argv[1]
    pool = mp.Pool(mp.cpu_count())

    resumes = []
    for root, directories, filenames in os.walk(resume_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            resumes.append((filename, filepath))

    results = [
        pool.apply(
            resume_result_wrapper,
            args=(resume[1],)
        ) for resume in resumes
    ]

    res_json = {}
    for resume, result in zip(resumes, results):
        res_json.update({
            resume[0]: result
        })
    with open(os.path.join(resume_dir, 'resumes.json'), 'w', encoding='utf-8') as fw:
        json.dump(res_json, fw, ensure_ascii=False, indent=2)