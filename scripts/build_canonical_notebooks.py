"""
Builder script to generate comprehensive, publication-grade Jupyter notebooks
for Balinese Mask Classification (Canonical 801 Leakage-Free Pipeline).
"""
import json
import os
from pathlib import Path

OUT_DIR = Path("notebooks_v2_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(code):
    lines = [line + "\n" for line in code.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

print("Helper functions defined successfully.")
