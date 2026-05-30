#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:26
@File    : execute_task.py
"""


from daao.actions import Action
from daao.schema import Message


class ExecuteTask(Action):
    name: str = "ExecuteTask"
    i_context: list[Message] = []

    async def run(self, *args, **kwargs):
        pass
