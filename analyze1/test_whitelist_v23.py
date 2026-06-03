#!/usr/bin/env python3
"""
v2.3 白名单逻辑验证脚本
构造包含F/W/Y的测试序列，全面覆盖白名单逻辑的各种边界情况和典型场景
"""
import json
import sys
from pathlib import Path

from csco_config import extract_cdr3_features
from csco_generator import check_hard_constraints, check_anti_patterns, load_strategy

STRATEGY_PATH = Path(__file__).parent / "output_server_v2.3" / "design_strategy.json"


def test_whitelist():
    strategy = load_strategy(str(STRATEGY_PATH))
    hc = strategy['hard_constraints']
    first_wl = hc.get('cdr3_first_residue_whitelist', [])
    last_wl = hc.get('cdr3_last_residue_whitelist', [])

    print("=" * 70)
    print("v2.3 白名单逻辑验证")
    print("=" * 70)
    print(f"首残基白名单: {first_wl}")
    print(f"尾残基白名单: {last_wl}")
    print(f"首残基芳香族要求: {hc.get('cdr3_min_aromatic_first', False)}")
    print(f"允许长度: {hc.get('cdr3_length_allowed', [])}")
    print()

    # 构造测试序列，覆盖各种边界情况
    test_cases = [
        # === 白名单内首残基 (F, W, Y) ===
        # F开头 - 允许长度
        ("FKDSPY", True, "F开头+允许长度6+尾Y在白名单"),
        ("FKDSPF", True, "F开头+允许长度6+尾F在白名单"),
        ("FHTSPPA", True, "F开头+允许长度7+尾A在白名单"),
        ("FHTSPR", False, "F开头+允许长度6+尾R不在白名单"),
        # W开头
        ("WTDKEY", True, "W开头+允许长度6+尾Y在白名单"),
        ("WVGDTPA", True, "W开头+允许长度7+尾A在白名单"),
        ("WTDKEK", False, "W开头+允许长度6+尾K不在白名单"),
        # Y开头
        ("YKNALY", True, "Y开头+允许长度6+尾Y在白名单"),
        ("YYNGQWY", True, "Y开头+允许长度7+尾Y在白名单"),
        ("YKNALK", False, "Y开头+允许长度6+尾K不在白名单"),

        # === 白名单外首残基 (应被过滤) ===
        ("AKDSPY", False, "A开头+不在白名单→应被过滤"),
        ("DKDSPY", False, "D开头+不在白名单→应被过滤"),
        ("GKDSPY", False, "G开头+不在白名单→应被过滤"),
        ("NKDSPY", False, "N开头+不在白名单→应被过滤(历史FC=0)"),
        ("IKDSPY", False, "I开头+不在白名单→应被过滤(历史FC=0)"),
        ("VKDSPY", False, "V开头+不在白名单→应被过滤(历史FC=0)"),
        ("SKDSPY", False, "S开头+不在白名单→应被过滤"),
        ("LKDSPY", False, "L开头+不在白名单→应被过滤"),

        # === 长度边界 ===
        ("FDY", False, "F开头+长度3+不在允许列表"),
        ("FWY", False, "F开头+长度3+不在允许列表"),
        ("FARTGQFTY", True, "F开头+长度9+尾Y在白名单"),
        ("FGSGARPSDYSY", True, "F开头+长度11+尾Y在白名单"),

        # === 尾残基边界 ===
        ("FKDSPC", False, "F开头+尾C不在白名单"),
        ("FKDSPM", False, "F开头+尾M不在白名单"),
        ("FKDSPG", False, "F开头+尾G不在白名单"),
        ("FKDSPQ", False, "F开头+尾Q不在白名单"),

        # === 反模式 ===
        ("FGGGDY", False, "F开头+含GGG反模式"),
        ("FSSSDY", False, "F开头+含SSS反模式"),
        ("FLLDY", False, "F开头+含LL反模式(但长度5不在允许列表)"),

        # === 芳香族首残基双重验证 ===
        ("FKDSPY", True, "F既是芳香族又在白名单→双重通过"),
        ("WKDSPY", True, "W既是芳香族又在白名单→双重通过"),
        ("YKDSPY", True, "Y既是芳香族又在白名单→双重通过"),
    ]

    passed = 0
    failed = 0
    results = []

    for cdr3, expected_pass, description in test_cases:
        ok, reason = check_hard_constraints(cdr3, strategy, verbose=True)
        has_anti, anti_pattern = check_anti_patterns(cdr3, strategy)

        # 如果有反模式，也算不通过
        actual_pass = ok and not has_anti
        match = actual_pass == expected_pass

        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        else:
            failed += 1

        result_line = f"  [{status}] {cdr3:15s} | 预期={'通过' if expected_pass else '拒绝':4s} | 实际={'通过' if actual_pass else '拒绝':4s} | {description}"
        if not ok:
            result_line += f" | 拒绝原因={reason}"
        if has_anti:
            result_line += f" | 反模式={anti_pattern}"
        results.append(result_line)

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    for r in results:
        print(r)

    print(f"\n总计: {len(test_cases)} | 通过: {passed} | 失败: {failed}")

    # 额外验证：确认所有白名单外首残基都被过滤
    print("\n" + "=" * 70)
    print("白名单覆盖性验证")
    print("=" * 70)
    from csco_config import AMINO_ACIDS, AROMATIC
    non_whitelist = [aa for aa in AMINO_ACIDS if aa not in first_wl]
    print(f"白名单内首残基: {first_wl} (共{len(first_wl)}个)")
    print(f"白名单外首残基: {non_whitelist} (共{len(non_whitelist)}个)")
    print(f"芳香族氨基酸: {sorted(AROMATIC)}")
    print(f"白名单与芳香族一致性: {set(first_wl) == AROMATIC}")

    # 验证白名单外首残基生成的序列全部被过滤
    filtered_count = 0
    for aa in non_whitelist:
        test_seq = f"{aa}KDSPY"  # 长度6, 尾Y在白名单
        ok, reason = check_hard_constraints(test_seq, strategy)
        if not ok:
            filtered_count += 1
            print(f"  ✓ {aa}开头 → 被过滤 (原因: {reason})")
        else:
            print(f"  ✗ {aa}开头 → 未被过滤! (白名单逻辑可能有误)")

    print(f"\n白名单外首残基过滤率: {filtered_count}/{len(non_whitelist)}")

    # 验证白名单内首残基生成的序列不被首残基规则过滤
    print("\n白名单内首残基通过率:")
    whitelist_pass = 0
    for aa in first_wl:
        test_seq = f"{aa}KDSPY"  # 长度6, 尾Y在白名单
        ok, reason = check_hard_constraints(test_seq, strategy)
        if ok:
            whitelist_pass += 1
            print(f"  ✓ {aa}开头 → 通过")
        else:
            print(f"  ✗ {aa}开头 → 被过滤 (原因: {reason})")

    print(f"白名单内首残基通过率: {whitelist_pass}/{len(first_wl)}")

    return failed == 0


if __name__ == "__main__":
    success = test_whitelist()
    sys.exit(0 if success else 1)
