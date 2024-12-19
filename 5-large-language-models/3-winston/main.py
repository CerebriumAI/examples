import json
import os

import requests
from huggingface_hub import login
from outlines import models, generate
from pydantic import BaseModel

login(token=os.environ.get("HF_AUTH_TOKEN"))
model = models.transformers("mistralai/Codestral-22B-v0.1")
boolean_generator = generate.choice(model, ["yes", "no"])
generator = generate.text(model)


class Item(BaseModel):
    data: dict


def predict(item):
    if item["action"] in ["opened", "synchronize"]:
        pr = item["pull_request"]
        repo_name = item["repository"]["full_name"]
        pr_number = item["number"]
        base_sha = pr["base"]["sha"]
        head_sha = pr["head"]["sha"]

        # Fetch the list of files changed
        files_changed = get_changed_files(repo_name, base_sha, head_sha)

        # Pass the files_changed data to your LLM processing function
        summary = process_files_with_llm(
            "You are a coding assistant reviewing the contents of a pull request on a Github repository. Based on the given code changes, give a high level summary of what the user has changed in bullet point format. Be informative and professional",
            files_changed,
        )
        leave__comment(repo_name, pr_number, summary)

        approval = boolean_generator(
            f"You are a coding assistant reviewing the contents of a pull request on a Github repository. I am providing you both the old and new code where deleted code is denoted by '-' and new code by '+' on the given code changes, do you you think the code changes look good to approve? If you think the PR is good to approve, respond 'yes' otherwise respond 'no'. If it is a complex PR then respond no even if it looks correct. Here is the code: {json.dumps(files_changed)}"
        )
        print(approval)

        if approval.strip().lower() == "yes":
            approve_pull_request(repo_name, pr_number, "Approved by Winston")
        else:
            comments = process_files_with_llm(
                "You are a coding assistant reviewing the contents of a pull request on a Github repository. I am providing you both the old and new code where deleted code is denoted by '-' and new code by '+' on the given code changes. Give feedback on the pull request of REQUIRED code corrections and why you think the user needs these corrections. Output the results with the filename and line number you are commenting on and the comment you have",
                files_changed=files_changed,
            )
            leave__comment(repo_name, pr_number, comments)

    return {"status": "success"}


def get_changed_files(repo_name, base_sha, head_sha):
    url = f"https://api.github.com/repos/{repo_name}/compare/{base_sha}...{head_sha}"
    headers = {"Authorization": f'token {os.environ.get("GITHUB_TOKEN")}'}
    response = requests.get(url, headers=headers)
    comparison = response.json()

    files_changed = []
    for file in comparison["files"]:
        file_info = {
            "filename": file["filename"],
            "status": file["status"],
            "changes": file["changes"],
            "patch": file.get("patch"),
        }

        files_changed.append(file_info)
    return files_changed


def process_files_with_llm(prompt, files_changed):
    return generator(f"{prompt}. Below is the code changes: {files_changed}")


def approve_pull_request(repo_name, pr_number, message):
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}/reviews"
    headers = {"Authorization": f'token {os.environ.get("GITHUB_TOKEN")}'}
    data = {"body": message, "event": "APPROVE"}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print(f"Pull request #{pr_number} approved successfully.")
    else:
        print(
            f"Failed to approve pull request #{pr_number}. Response: {response.content}"
        )


def leave__comment(repo_name, pr_number, comment):
    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    headers = {"Authorization": f'token {os.environ.get("GITHUB_TOKEN")}'}
    data = {"body": comment}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f"Comment added to pull request #{pr_number} successfully.")
    else:
        print(
            f"Failed to add comment to pull request #{pr_number}. Response: {response.content}"
        )
