import glob
import re

def insert_name_in_html(file_path, name, github_repo_link, github_profile_link):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    insert_html = (f'<br><h2>Generated by {name} for <a href="{github_repo_link}" '
                   f'>AutoDiff</a>. '
                   f'<a href="{github_profile_link}">(GitHub Profile)</a></h2>')

    pattern = r'(</h1>)'
    content = re.sub(pattern, r'\1' + insert_html, content, flags=re.DOTALL)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

if __name__ == "__main__":
    name = "Amelia Li"
    github_repo_link = "https://github.com/amelialwx/AutoDiff"
    github_profile_link = "https://github.com/amelialwx"
    html_files = glob.glob('tests/htmlcov/*.html')

    for html_file in html_files:
        insert_name_in_html(html_file, name, github_repo_link, github_profile_link)
