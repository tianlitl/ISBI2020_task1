#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import CONFIG
from train import train, evaluate

def main():
    record_epochs, accs, losses = train(CONFIG)
    
if __name__ == '__main__':
    main()