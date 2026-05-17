src2 = open('/notebooks/PMamba/experiments/utils/parameters.py').read()
old2 = """    parser.add_argument(
        '--pts-freeze-after-ep',"""
new2 = """    parser.add_argument(
        '--pts-extra-target',
        type=int,
        default=None,
        help='extend ramp past 256 to this target after ep 100')
    parser.add_argument(
        '--pts-extra-epochs',
        type=int,
        default=None,
        help='epochs to spend ramping 256 -> pts_extra_target after ep 100')
    parser.add_argument(
        '--pts-freeze-after-ep',"""
assert old2 in src2
open('/notebooks/PMamba/experiments/utils/parameters.py','w').write(src2.replace(old2, new2, 1))
print('parameters.py patched')
