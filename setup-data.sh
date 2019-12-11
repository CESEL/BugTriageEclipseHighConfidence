#!/bin/bash
unzip resources/eclipse_all_bug_comments.zip -d resources
git clone https://github.com/eclipse/eclipse.jdt.core.git
git clone https://github.com/eclipse/eclipse.jdt.debug.git
git clone https://github.com/eclipse/eclipse.jdt.ui.git
git clone https://github.com/eclipse/eclipse.platform.common.git
git clone https://github.com/eclipse/eclipse.platform.debug.git
git clone https://github.com/eclipse/eclipse.platform.releng.git
git clone https://github.com/eclipse/eclipse.platform.releng.aggregator.git
git clone https://github.com/eclipse/eclipse.platform.releng.buildtools.git
git clone https://github.com/eclipse/eclipse.platform.resources.git
git clone https://github.com/eclipse/eclipse.platform.runtime.git
git clone https://github.com/eclipse/eclipse.platform.swt.git
git clone https://github.com/eclipse/eclipse.platform.team.git
git clone https://github.com/eclipse/eclipse.platform.text.git
git clone https://github.com/eclipse/eclipse.platform.ua.git
git clone https://github.com/eclipse/eclipse.platform.ui.git
python data_collection/past_bug_history_collector.py
