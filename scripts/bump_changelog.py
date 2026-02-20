#!/usr/bin/env python3
"""
Changelog Generator using Claude API

This script analyzes git changes since the last release and uses Claude API
to generate meaningful changelog entries for CHANGELOG.md

Requirements:
- pip install anthropic gitpython
- Set ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import git
from anthropic import Anthropic


class ChangelogGenerator:
    def __init__(self, repo_path: str = ".", api_key: Optional[str] = None):
        self.repo_path = Path(repo_path).resolve()
        self.repo = git.Repo(repo_path)
        
        # Initialize Claude API
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        
        self.client = Anthropic(api_key=api_key)
        self.changelog_path = self.repo_path / "CHANGELOG.md"
    
    def get_last_release_tag(self) -> Optional[str]:
        """Get the most recent release tag."""
        try:
            # Get all tags sorted by version
            tags = self.repo.git.tag("--sort=-version:refname").split('\n')
            # Filter for version-like tags (e.g., v1.0.0, 1.0.0)
            version_pattern = re.compile(r'^v?\d+\.\d+\.\d+')
            for tag in tags:
                if tag.strip() and version_pattern.match(tag.strip()):
                    return tag.strip()
            return None
        except git.exc.GitCommandError:
            return None
    
    def get_commits_since_tag(self, tag: Optional[str]) -> List[git.Commit]:
        """Get all commits since the specified tag."""
        if tag:
            try:
                commits = list(self.repo.iter_commits(f"{tag}..HEAD"))
                return commits
            except git.exc.GitCommandError:
                print(f"Warning: Tag '{tag}' not found, using all commits")
                return list(self.repo.iter_commits("HEAD"))
        else:
            # If no tag found, get last 50 commits
            return list(self.repo.iter_commits("HEAD", max_count=50))
    
    def get_file_changes(self, commits: List[git.Commit]) -> List[Tuple[str, str, str]]:
        """Get detailed file changes for the commits."""
        changes = []
        
        for commit in commits:
            if commit.parents:  # Skip initial commit
                # Get diff for this commit
                diffs = commit.parents[0].diff(commit, create_patch=True)
                
                for diff in diffs:
                    if diff.a_path:
                        change_type = "modified"
                        if diff.new_file:
                            change_type = "added"
                        elif diff.deleted_file:
                            change_type = "deleted"
                        elif diff.renamed_file:
                            change_type = "renamed"
                        
                        # Get a sample of the actual changes (first 500 chars)
                        diff_text = str(diff)[:500] if diff else ""
                        
                        changes.append((
                            diff.a_path or diff.b_path,
                            change_type,
                            diff_text
                        ))
        
        return changes
    
    def analyze_changes_with_claude(self, commits: List[git.Commit], 
                                  file_changes: List[Tuple[str, str, str]]) -> str:
        """Use Claude API to analyze changes and generate changelog entries."""
        
        # Get repository URL for GitHub links
        repo_url = self.get_github_repo_url()
        
        # Prepare commit information for Claude analysis
        commit_details = []
        for commit in commits:
            commit_details.append({
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:7],
                "message": commit.message.strip(),
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "date": datetime.fromtimestamp(commit.committed_date).isoformat(),
                "files": list(commit.stats.files.keys()) if commit.parents else []
            })
        
        prompt = f"""
Please analyze the following git commits and generate improved, user-friendly descriptions for each commit.

REPOSITORY: {repo_url or 'Unknown'}
COMMITS TO ANALYZE ({len(commits)} total):

"""
        
        for commit in commit_details:
            prompt += f"""
Commit: {commit['short_hash']}
Original message: {commit['message']}
Author: {commit['author_name']}
Files changed: {', '.join(commit['files'][:5])}{'...' if len(commit['files']) > 5 else ''}

"""
        
        prompt += """
For each commit, please provide an improved description that:
1. Starts with the nature of change (Add, Fix, Update, Remove, Refactor, etc.)
2. Includes BOTH the type of change AND specific details about what was changed
3. Is clear and user-facing (not overly technical)
4. Keeps it concise but informative (one line, but can be longer if needed)
5. Provides enough context so someone can understand what specifically was done

Examples of good descriptions:
- Add configuration option for custom timeout values in network requests
- Fix memory leak in data processing pipeline when handling large files
- Update dependency validation to support newer package versions
- Remove deprecated API endpoints for user authentication
- Refactor database connection logic to improve error handling

Please respond with ONLY the improved descriptions, one per line, in the same order as the commits above.
Do not include commit hashes or any other formatting - just the description text.
"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response and combine with commit info
            improved_descriptions = [desc.strip() for desc in response.content[0].text.strip().split('\n') if desc.strip()]
            
            # Generate formatted changelog entries
            changelog_entries = []
            for i, commit in enumerate(commit_details):
                description = improved_descriptions[i] if i < len(improved_descriptions) else commit['message'].split('\n')[0]
                
                # Get GitHub username from email or use name
                author_username = self.get_github_username(commit['author_email'], commit['author_name'])
                
                if repo_url:
                    commit_link = f"[`{commit['short_hash']}`]({repo_url}/commit/{commit['hash']})"
                else:
                    commit_link = f"`{commit['short_hash']}`"
                
                entry = f"- {commit_link} - {description} *(commit by [@{author_username}](https://github.com/{author_username}))*"
                changelog_entries.append(entry)
            
            return '\n'.join(changelog_entries)
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return self._generate_fallback_changelog(commits)
    
    def get_github_repo_url(self) -> Optional[str]:
        """Extract GitHub repository URL from git remotes."""
        try:
            origin = self.repo.remote('origin')
            url = origin.url
            
            # Convert SSH URL to HTTPS if needed
            if url.startswith('git@github.com:'):
                url = url.replace('git@github.com:', 'https://github.com/')
            if url.endswith('.git'):
                url = url[:-4]
                
            return url
        except:
            return None
    
    def get_github_username(self, email: str, name: str) -> str:
        """Extract or guess GitHub username from email/name."""
        # Try to extract from email
        if '@' in email:
            username = email.split('@')[0]
            # Clean up common email patterns
            username = username.replace('.', '').replace('-', '').replace('_', '')
            if username.isalnum():
                return username
        
        # Fallback to name (cleaned up)
        username = name.lower().replace(' ', '').replace('.', '').replace('-', '')
        return username if username else 'unknown'
    
    def _generate_fallback_changelog(self, commits: List[git.Commit]) -> str:
        """Generate a basic changelog if Claude API fails."""
        changelog_entries = []
        repo_url = self.get_github_repo_url()
        
        for commit in commits:
            author_username = self.get_github_username(commit.author.email, commit.author.name)
            
            if repo_url:
                commit_link = f"[`{commit.hexsha[:7]}`]({repo_url}/commit/{commit.hexsha})"
            else:
                commit_link = f"`{commit.hexsha[:7]}`"
            
            # Clean up commit message
            message = commit.message.strip().split('\n')[0]
            
            entry = f"- {commit_link} - {message} *(commit by [@{author_username}](https://github.com/{author_username}))*"
            changelog_entries.append(entry)
        
        return '\n'.join(changelog_entries)
    
    def update_changelog(self, new_entry: str, version: str = None):
        """Update the CHANGELOG.md file with new entries."""
        if not version:
            version = "Unreleased"
        
        # Read existing changelog
        if self.changelog_path.exists():
            with open(self.changelog_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Create new entry
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_section = f"## [{version}] - {date_str}\n\n{new_entry}\n\n"
        
        # Insert new entry after the header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('## '):
                header_end = i
                break
            elif i > 10:  # Safety check
                header_end = len(lines)
                break
        
        if header_end == 0:
            # No existing entries, add after initial header
            for i, line in enumerate(lines):
                if line.strip() == '' and i > 0:
                    header_end = i + 1
                    break
        
        # Insert new content
        lines.insert(header_end, new_section.rstrip())
        
        # Write back to file
        with open(self.changelog_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n\n')
        
        print(f"Updated {self.changelog_path}")
    
    def run(self, version: str = None, auto_update: bool = False):
        """Main execution function."""
        print("Analyzing repository changes...")
        
        # Get last release tag
        last_tag = self.get_last_release_tag()
        if last_tag:
            print(f"Last release tag: {last_tag}")
        else:
            print("No release tags found, analyzing recent commits")
        
        # Get commits since last release
        commits = self.get_commits_since_tag(last_tag)
        if not commits:
            print("No new commits found")
            return
        
        print(f"Found {len(commits)} commits since last release")
        
        # Get file changes
        print("Analyzing file changes...")
        file_changes = self.get_file_changes(commits)
        
        # Generate changelog with Claude
        print("Generating changelog with Claude...")
        changelog_entry = self.analyze_changes_with_claude(commits, file_changes)
        
        print("\n" + "="*60)
        print("GENERATED CHANGELOG ENTRY:")
        print("="*60)
        print(changelog_entry)
        print("="*60)
        
        # Update changelog with or without confirmation
        if auto_update:
            print("\nAuto-updating CHANGELOG.md...")
            self.update_changelog(changelog_entry, version)
        else:
            response = input("\nDo you want to add this to CHANGELOG.md? (y/n): ").lower()
            if response in ['y', 'yes']:
                self.update_changelog(changelog_entry, version)
            else:
                print("Changelog not updated")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate changelog entries using Claude API")
    parser.add_argument("--repo", default=".", help="Path to git repository")
    parser.add_argument("--version", help="Version number for changelog entry")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-update changelog without confirmation")
    
    args = parser.parse_args()
    
    try:
        generator = ChangelogGenerator(args.repo, args.api_key)
        generator.run(args.version, args.yes)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()