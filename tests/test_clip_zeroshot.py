from src.clip_zeroshot import build_prompts


def test_build_prompts_emits_one_template_per_class():
    class_names = ["Abyssinian", "Boxer", "Persian"]
    prompts = build_prompts(class_names, templates=["a photo of a {}, a type of pet."])
    assert prompts == [
        "a photo of a abyssinian, a type of pet.",
        "a photo of a boxer, a type of pet.",
        "a photo of a persian, a type of pet.",
    ]


def test_prompt_ensemble_multiple_templates():
    class_names = ["Abyssinian"]
    prompts = build_prompts(class_names,
                            templates=["a photo of a {}.", "a picture of a {} pet."])
    assert len(prompts) == 2


def test_underscore_to_space_in_class_name():
    prompts = build_prompts(["Maine_Coon"], templates=["a photo of a {}."])
    assert prompts == ["a photo of a maine coon."]
