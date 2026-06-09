"""Backbone shells must raise a clear error until they're wired up.

This protects against silently shipping a half-implemented matcher.
"""

import pytest

from cma.matchers import ELoFTRMatcher, MatcherNotInstalled


@pytest.mark.parametrize("cls", [ELoFTRMatcher])
def test_shells_raise_until_implemented(cls):
    with pytest.raises(MatcherNotInstalled):
        cls()
