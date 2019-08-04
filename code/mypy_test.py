#!/usr/bin/env python3
import my_module


def thing(a: str) -> str:
    print("thing string {}".format(a))
    return "answer"


thing("james")
# thing(1)
my_module.other_thing("anthony")
# my_module.other_thing(2)
