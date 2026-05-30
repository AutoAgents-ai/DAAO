#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:35
@File    : __init__.py
"""

from enum import Enum
from daao.tools import libs  # this registers all tools
from daao.tools.tool_registry import TOOL_REGISTRY

_ = libs, TOOL_REGISTRY  # Avoid pre-commit error


class SearchEngineType(Enum):
    SERPAPI_GOOGLE = "serpapi"
    SERPER_GOOGLE = "serper"
    DIRECT_GOOGLE = "google"
    DUCK_DUCK_GO = "ddg"
    CUSTOM_ENGINE = "custom"
    BING = "bing"


class WebBrowserEngineType(Enum):
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    CUSTOM = "custom"

    @classmethod
    def __missing__(cls, key):
        """Default type conversion"""
        return cls.CUSTOM


class SearchInterface:
    async def asearch(self, *args, **kwargs):
        ...
