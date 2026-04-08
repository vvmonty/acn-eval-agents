# Food Planner Agent — Evaluation Analysis Report

**Dataset:** `FoodPlannerEval`  
**Experiment Run:** `food_planner_v1_full_eval - 2026-04-08T19:59:16.336470Z`  
**Generated:** 2026-04-08 20:19  
**Total Items:** 30

---

## 1. Overall Score Summary

| Metric | Type | Pass Rate | Valid Items |
|---|---|---|---|
| `required_tools` | Rule-based | 40.0% | 30/30 |
| `output_present` | Rule-based | 96.7% | 30/30 |
| `shopping_list` | Rule-based | 90.0% | 30/30 |
| `dietary_compliance` | LLM-judge | 96.7% | 30/30 |
| `time_constraint` | LLM-judge | 80.0% | 30/30 |
| `recipe_present` | LLM-judge | 70.0% | 30/30 |
| `actionable_steps` | LLM-judge | 70.0% | 30/30 |
| `recipe_present_strict` | LLM-judge | 26.7% | 30/30 |
| `actionable_steps_strict` | LLM-judge | 33.3% | 30/30 |
| `recipe_present_lenient` | LLM-judge | 73.3% | 30/30 |
| `actionable_steps_lenient` | LLM-judge | 73.3% | 30/30 |
| `serving_size` | LLM-judge | 73.3% | 30/30 |

---

## 2. Pass Rates by Test Category

| Category | N | required | output_p | shopping | dietary_ | time_con | recipe_p | actionab | recipe_p | actionab | recipe_p | actionab | serving_ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unknown | 30 | 40% | 97% | 90% | 97% | 80% | 70% | 70% | 27% | 33% | 73% | 73% | 73% |
| **OVERALL** | **30** | **40%** | **97%** | **90%** | **97%** | **80%** | **70%** | **70%** | **27%** | **33%** | **73%** | **73%** | **73%** |

---

## 3. Failure Analysis

**17 items** failed ≥ 2 LLM-judge metrics:

| Item | Category | # Failures | dietary | time | recipe | steps | recipe_s | steps_s | recipe_l | steps_l | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FoodPlannerEval:9de81c336d9e44b04ad08423a0738c05cc4263d59d4c37fb079789568fc02620 | unknown | 8 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:7e227693bb297a96a16593a421495dc3157bcfeaf063511bb7a3ad70ef789b6b | unknown | 8 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:e0f39a096b09be7e8c9b1ff44a38c2e6ab3040e76683c9b801958a69f8d3941e | unknown | 8 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:cc9a34d761823b579c4d2a007de9eeae411aeab891d9a6f4c3e0c26dbe42f924 | unknown | 7 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:2a46d53d0cb1a6d50feb9e8e087ce727169a539d58e0505db15f824824b60c0e | unknown | 7 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:688efca09780268f9013f6edc88203c5c9f4340a47a9218df6d3e43ea6913d52 | unknown | 7 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| FoodPlannerEval:4b90acf0c2dd35695482e7d5ac3ad5893ec47e32362cb2f9ca9000d2868af18c | unknown | 7 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:4ebebc08dcbe986ae492c7d341687013abb1c2a9f6df26f58a299d041b733f5a | unknown | 7 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FoodPlannerEval:689b8222c4c1a2a42a14edfa922f0132f1204c22bd2481c709bde64e90f40ab8 | unknown | 4 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:388a02fd427376ce3cf003915b9bb8f562565f90100e9f42b5c1d1aa043f491d | unknown | 3 | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:ee622327b1b1797aab8d36d57a29601f983eb0c6f089abb95537437762e075a4 | unknown | 3 | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:6e72a6d3c1abb56ee83f9b6ac854adbd0a96777d6a44af7950823f8a4cb08a00 | unknown | 3 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| FoodPlannerEval:0f8db471dfaf11ad5647f62d4e50c9b4e49b953b616460dc1bbbd79f383f1eeb | unknown | 2 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:158c149aa4710b27d055c27b7335e7689b49f482f81228a2b1d76ce31036b24c | unknown | 2 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:ab64b4ddec60b98dabce43c8cac4c889f5d56a6eff9debbcf83b797d6f9b6102 | unknown | 2 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:dec62cb9729a76907575a2f2b52f0e7b4702c61c78de1e83b0cf587135f53c84 | unknown | 2 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| FoodPlannerEval:f1264c21d79d8c2d6c272bc33e001c7241435086166113ca9553003052080763 | unknown | 2 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |

---

## 4. Key Observations

- **Weakest metric**: `recipe_present_strict` (27% pass rate). **Strongest**: `output_present` (97%).
- **Hardest test category**: `unknown` (lowest avg LLM-judge pass rate). **Easiest**: `unknown`.
- **Tool trajectory**: 18 item(s) missed at least one required tool call. Review those items in Langfuse for skipped CFIA checks or missing modify_recipe calls.

---

## 5. Items by Test Category

### unknown (30 items)

| Item | Ingredient | Time | Srv | requir | output | shoppi | dietar | time_c | recipe | action | recipe | action | recipe | action | servin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FoodPlannerEval:0f8db471dfaf11ad5647f62d4e50c9b4e49b953b616460dc1bbbd79f383f1eeb |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:158c149aa4710b27d055c27b7335e7689b49f482f81228a2b1d76ce31036b24c |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:2a46d53d0cb1a6d50feb9e8e087ce727169a539d58e0505db15f824824b60c0e |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:2a9c0c45004062644f53298192cd9db1e07c94f490737ed7b0544e7d3b99c3c1 |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:388a02fd427376ce3cf003915b9bb8f562565f90100e9f42b5c1d1aa043f491d |  | Nonem | None | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:4b90acf0c2dd35695482e7d5ac3ad5893ec47e32362cb2f9ca9000d2868af18c |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:4ebebc08dcbe986ae492c7d341687013abb1c2a9f6df26f58a299d041b733f5a |  | Nonem | None | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:5ef164c046fd239f0c9c6d1bf87797d436adccbe9a37d414baa85c710c8a4e09 |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:64fa8d33c677efe3e4756371e1deff713c6670bb68db6b34d64f04680c2811ef |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:688efca09780268f9013f6edc88203c5c9f4340a47a9218df6d3e43ea6913d52 |  | Nonem | None | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| FoodPlannerEval:689b8222c4c1a2a42a14edfa922f0132f1204c22bd2481c709bde64e90f40ab8 |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:6e72a6d3c1abb56ee83f9b6ac854adbd0a96777d6a44af7950823f8a4cb08a00 |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| FoodPlannerEval:720f6e60b27a9ca9863f933791825fcd9e00f6464e3d074948bc986fe748346d |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:7ada4e17b31599ca6da158a7e6a855a44376efb7391de26896e0f9147389077f |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:7e227693bb297a96a16593a421495dc3157bcfeaf063511bb7a3ad70ef789b6b |  | Nonem | None | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:88870ca6e13fdd22b8706ba4ed19ff0fe1ceb99d9c08461df023331d11c11a7e |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:9aa881f589cb282e59341e3e6eee3a983e7845975a1978faed7edaeba602abd7 |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:9de81c336d9e44b04ad08423a0738c05cc4263d59d4c37fb079789568fc02620 |  | Nonem | None | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:ab64b4ddec60b98dabce43c8cac4c889f5d56a6eff9debbcf83b797d6f9b6102 |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:ab9d7e446599804e1a5b2f30907a3597ad19a51f4a73bb336c97ba7fca50f0f9 |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:b52f0b066c90a34ff7a8c39a3b52adaf5051dd229699134e54dd97086ecdb6bc |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:cc9a34d761823b579c4d2a007de9eeae411aeab891d9a6f4c3e0c26dbe42f924 |  | Nonem | None | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:ce1615ee3253eff2f26951727a7e433885c0320c06ed173e15a791641b2088e4 |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| FoodPlannerEval:d44d9d9dc5b6e2269fc83976309b35c0cd70371832e6d7588c664287260c3d6f |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:dec62cb9729a76907575a2f2b52f0e7b4702c61c78de1e83b0cf587135f53c84 |  | Nonem | None | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:e0f39a096b09be7e8c9b1ff44a38c2e6ab3040e76683c9b801958a69f8d3941e |  | Nonem | None | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FoodPlannerEval:e6ae5a811d704a234c422d81531af28714dfc046c8b294d7b3f1b43f8cdfcf34 |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:ee622327b1b1797aab8d36d57a29601f983eb0c6f089abb95537437762e075a4 |  | Nonem | None | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:f1264c21d79d8c2d6c272bc33e001c7241435086166113ca9553003052080763 |  | Nonem | None | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| FoodPlannerEval:f6cfce838b9189798a0ff43ac3ae539de31634a38ab88621ffcf4e9ace547c4b |  | Nonem | None | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

---

*Generated by EVAL team Phase 5 analysis script.*