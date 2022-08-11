import sys
import os
from os import path
import subprocess
import configparser
from setuptools import setup, find_packages, Command

# Get config parameters
config = configparser.ConfigParser()
config.read('setup.cfg')
pkg_name = config['metadata']['name']
pypi_server = config['netsquid']['pypi-server']
docs_server = "docs.netsquid.org"


def load_readme_text():
    """Load in README file as a string."""
    try:
        dir_path = path.abspath(path.dirname(__file__))
        with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_requirements():
    """Load in requirements.txt as a list of strings."""
    try:
        dir_path = path.abspath(path.dirname(__file__))
        with open(path.join(dir_path, 'requirements.txt'), encoding='utf-8') as f:
            install_requires = [line.strip() for line in f.readlines()]
            return install_requires
    except FileNotFoundError:
        return ""


class DeployCommand(Command):
    """Run command for uploading binary wheel files to NetSquid PyPi index.

    """
    description = "Deploy binary wheel files to NetSquid PyPi index."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("Uploading binary snippet {} wheels to {} (requires authentication)"
              .format(pkg_name, pypi_server))
        if 'NETSQUIDCI_USER' not in os.environ:
            sys.exit("ERROR: environment variable NETSQUIDCI_USER is not defined.")
        if 'NETSQUIDCI_BASEDIR' not in os.environ:
            sys.exit("ERROR: environment variable NETSQUIDCI_BASEDIR is not defined.")
        # Upload wheel files
        pkg_dir = f"{os.environ['NETSQUIDCI_BASEDIR']}/pypi/{pkg_name}/"
        sftp_args = ["/usr/bin/sftp"]
        if 'NETSQUIDCI_PROXY' in os.environ:
            # sftp_args += ["-J", f"{os.environ['NETSQUIDCI_PROXY']}"]
            # Compatible with older openssh clients:
            sftp_args += ["-o", f"ProxyCommand=/usr/bin/ssh -W %h:%p {os.environ['NETSQUIDCI_PROXY']}"]
        sftp_args.append(f"{os.environ['NETSQUIDCI_USER']}@{pypi_server}")
        if len([f for f in os.listdir("dist/") if f.endswith(".whl")]) > 0:
            subprocess.run(
                sftp_args, input="put -p dist/*.whl {}".format(pkg_dir).encode()).check_returncode()
        else:
            sys.exit("ERROR: no wheel files in dist/ to upload.")


class DeployDocsCommand(Command):
    """Run command for uploading documentation files to NetSquid doc server.

    Requires authentication.
    Requires documentation to have been built.

    """
    description = f"Deploy documentation files to Snippet documentation server {docs_server}."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if 'NETSQUIDCI_USER' not in os.environ:
            sys.exit("ERROR: environment variable NETSQUIDCI_USER is not defined.")
        if 'NETSQUIDCI_BASEDIR' not in os.environ:
            sys.exit("ERROR: environment variable NETSQUIDCI_BASEDIR is not defined.")
        sftp_args = ["/usr/bin/sftp"]
        if 'NETSQUIDCI_PROXY' in os.environ:
            # sftp_args += ["-J", f"{os.environ['NETSQUIDCI_PROXY']}"]
            # Compatible with older openssh clients:
            sftp_args += ["-o", f"ProxyCommand=/usr/bin/ssh -W %h:%p {os.environ['NETSQUIDCI_PROXY']}"]
        sftp_args.append(f"{os.environ['NETSQUIDCI_USER']}@{docs_server}")
        docs_dir = f"{os.environ['NETSQUIDCI_BASEDIR']}/docs/snippets/{pkg_name}/"
        # Create directory if it doesn't exist:
        subprocess.run(sftp_args, input=f"mkdir {docs_dir}".encode()).check_returncode()
        # Upload docs:
        subprocess.run(sftp_args, input=f"put -pR docs/build/html/. {docs_dir}".encode()).check_returncode()


setup(
    cmdclass={"deploy": DeployCommand,
              "deploy_docs": DeployDocsCommand},
    long_description=load_readme_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    packages=find_packages(exclude=('tests', 'docs', 'examples')),  # if offering a package
    # py_modules=['pkgname.replace('-', '_')'],  # if offering a single module file
    install_requires=load_requirements(),
    test_suite=pkg_name.replace('-', '_'),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
)
