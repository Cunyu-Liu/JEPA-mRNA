import hashlib


def add_clean_speed_args(parser):
    parser.add_argument("--clean-speed-protocol", action="store_true")
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-count", type=int)


def validate_clean_speed_rows(rows, args):
    if not args.clean_speed_protocol:
        return False
    if args.smoke or args.limit is not None or args.clean_audit is not None:
        raise ValueError("clean speed mode cannot be combined with smoke, limit or a different clean audit")
    if args.repeats != 3 or not args.expected_data_sha256 or args.expected_count != len(rows) or len(rows) < 2:
        raise ValueError("clean speed mode requires three full repeats and the registered nonzero raw count")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != args.expected_data_sha256:
        raise ValueError("clean speed data identity mismatch")
    return True
