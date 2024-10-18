import re
from collections import defaultdict

# Define regex patterns for definitions, lemmas, and theorems
definition_pattern = re.compile(r'^(def|abbreviation)\s+(\S+)\s*[:\(]', re.MULTILINE)
lemma_pattern = re.compile(r'^(lemma|theorem|@[^\n]*\s+(lemma|theorem))\s+(\S+)\s*', re.MULTILINE)
comment_pattern = re.compile(r'/-(?:.|\n)*?-/')  # To remove multiline comments

# Read the Lean file content
lean_file_content = """
/-
Copyright (c) 2014 Jeremy Avigad. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jeremy Avigad, Leonardo de Moura, Floris van Doorn, Amelia Livingston, Yury Kudryashov,
Neil Strickland, Aaron Anderson
-/

import algebra.hom.group

/-!
# Divisibility

> THIS FILE IS SYNCHRONIZED WITH MATHLIB4.
> Any changes to this file require a corresponding PR to mathlib4.

This file defines the basics of the divisibility relation in the context of `(comm_)` `monoid`s.

## Main definitions

 * `monoid.has_dvd`

## Implementation notes

The divisibility relation is defined for all monoids, and as such, depends on the order of
  multiplication if the monoid is not commutative. There are two possible conventions for
  divisibility in the noncommutative context, and this relation follows the convention for ordinals,
  so `a | b` is defined as `∃ c, b = a * c`.

## Tags

divisibility, divides
-/

variables {α : Type*}

section semigroup

variables [semigroup α] {a b c : α}

/-- There are two possible conventions for divisibility, which coincide in a `comm_monoid`.
    This matches the convention for ordinals. -/
@[priority 100]
instance semigroup_has_dvd : has_dvd α :=
has_dvd.mk (λ a b, ∃ c, b = a * c)

-- TODO: this used to not have `c` explicit, but that seems to be important
--       for use with tactics, similar to `exists.intro`
theorem dvd.intro (c : α) (h : a * c = b) : a ∣ b :=
exists.intro c h^.symm

alias dvd.intro ← dvd_of_mul_right_eq

theorem exists_eq_mul_right_of_dvd (h : a ∣ b) : ∃ c, b = a * c := h

theorem dvd.elim {P : Prop} {a b : α} (H₁ : a ∣ b) (H₂ : ∀ c, b = a * c → P) : P :=
exists.elim H₁ H₂

local attribute [simp] mul_assoc mul_comm mul_left_comm

@[trans] theorem dvd_trans : a ∣ b → b ∣ c → a ∣ c
| ⟨d, h₁⟩ ⟨e, h₂⟩ := ⟨d * e, h₁ ▸ h₂.trans $ mul_assoc a d e⟩

alias dvd_trans ← has_dvd.dvd.trans

instance : is_trans α (∣) := ⟨λ a b c, dvd_trans⟩

@[simp] theorem dvd_mul_right (a b : α) : a ∣ a * b := dvd.intro b rfl

theorem dvd_mul_of_dvd_left (h : a ∣ b) (c : α) : a ∣ b * c :=
h.trans (dvd_mul_right b c)

alias dvd_mul_of_dvd_left ← has_dvd.dvd.mul_right

theorem dvd_of_mul_right_dvd (h : a * b ∣ c) : a ∣ c :=
(dvd_mul_right a b).trans h

section map_dvd

variables {M N : Type*} [monoid M] [monoid N]

lemma map_dvd {F : Type*} [mul_hom_class F M N] (f : F) {a b} : a ∣ b → f a ∣ f b
| ⟨c, h⟩ := ⟨f c, h.symm ▸ map_mul f a c⟩

lemma mul_hom.map_dvd (f : M →ₙ* N) {a b} : a ∣ b → f a ∣ f b := map_dvd f

lemma monoid_hom.map_dvd (f : M →* N) {a b} : a ∣ b → f a ∣ f b := map_dvd f

end map_dvd

end semigroup

section monoid

variables [monoid α] {a b : α}

@[refl, simp] theorem dvd_refl (a : α) : a ∣ a := dvd.intro 1 (mul_one a)
theorem dvd_rfl : ∀ {a : α}, a ∣ a := dvd_refl
instance : is_refl α (∣) := ⟨dvd_refl⟩

theorem one_dvd (a : α) : 1 ∣ a := dvd.intro a (one_mul a)

lemma dvd_of_eq (h : a = b) : a ∣ b := by rw h

alias dvd_of_eq ← eq.dvd

end monoid

section comm_semigroup

variables [comm_semigroup α] {a b c : α}

theorem dvd.intro_left (c : α) (h : c * a = b) : a ∣ b :=
dvd.intro _ (begin rewrite mul_comm at h, apply h end)

alias dvd.intro_left ← dvd_of_mul_left_eq

theorem exists_eq_mul_left_of_dvd (h : a ∣ b) : ∃ c, b = c * a :=
dvd.elim h (assume c, assume H1 : b = a * c, exists.intro c (eq.trans H1 (mul_comm a c)))

lemma dvd_iff_exists_eq_mul_left : a ∣ b ↔ ∃ c, b = c * a :=
⟨exists_eq_mul_left_of_dvd, by { rintro ⟨c, rfl⟩, exact ⟨c, mul_comm _ _⟩, }⟩

theorem dvd.elim_left {P : Prop} (h₁ : a ∣ b) (h₂ : ∀ c, b = c * a → P) : P :=
exists.elim (exists_eq_mul_left_of_dvd h₁) (assume c, assume h₃ : b = c * a, h₂ c h₃)

@[simp] theorem dvd_mul_left (a b : α) : a ∣ b * a := dvd.intro b (mul_comm a b)

theorem dvd_mul_of_dvd_right (h : a ∣ b) (c : α) : a ∣ c * b :=
begin rw mul_comm, exact h.mul_right _ end

alias dvd_mul_of_dvd_right ← has_dvd.dvd.mul_left

local attribute [simp] mul_assoc mul_comm mul_left_comm

theorem mul_dvd_mul : ∀ {a b c d : α}, a ∣ b → c ∣ d → a * c ∣ b * d
| a ._ c ._ ⟨e, rfl⟩ ⟨f, rfl⟩ := ⟨e * f, by simp⟩

theorem dvd_of_mul_left_dvd (h : a * b ∣ c) : b ∣ c :=
dvd.elim h (λ d ceq, dvd.intro (a * d) (by simp [ceq]))

end comm_semigroup

section comm_monoid

variables [comm_monoid α] {a b : α}

theorem mul_dvd_mul_left (a : α) {b c : α} (h : b ∣ c) : a * b ∣ a * c :=
mul_dvd_mul (dvd_refl a) h

theorem mul_dvd_mul_right (h : a ∣ b) (c : α) : a * c ∣ b * c :=
mul_dvd_mul h (dvd_refl c)

theorem pow_dvd_pow_of_dvd {a b : α} (h : a ∣ b) : ∀ n : ℕ, a ^ n ∣ b ^ n
| 0     := by rw [pow_zero, pow_zero]
| (n+1) := by { rw [pow_succ, pow_succ], exact mul_dvd_mul h (pow_dvd_pow_of_dvd n) }

end comm_monoid

"""

# Remove comments to simplify parsing
lean_file_content = re.sub(comment_pattern, '', lean_file_content)

# Extract definitions
definitions = definition_pattern.findall(lean_file_content)
definitions = [name for (_, name) in definitions]

# Extract lemmas and theorems
lemmas = lemma_pattern.findall(lean_file_content)
lemmas = [name for (_, _, name) in lemmas]

# Combine all nodes
nodes = definitions + lemmas

# Initialize relationships
dependencies = defaultdict(set)
equivalences = set()
affiliations = set()
opposites = set()
synonyms = set()
properties = set()

# Function to extract dependencies from a lemma or definition
def extract_dependencies(name, content):
    deps = set()
    # Simple regex to find names of other definitions/lemmas used
    # This is a naive approach and may need to be refined
    for node in nodes:
        if node != name and re.search(r'\b' + re.escape(node) + r'\b', content):
            deps.add(node)
    return deps

# Split the content into individual declarations
declarations = re.split(r'\n\s*\n', lean_file_content)

for decl in declarations:
    # Find the name of the current declaration
    name_match = re.match(r'^(def|lemma|theorem|abbreviation)\s+(\S+)', decl)
    if name_match:
        name = name_match.group(2)
        # Extract dependencies
        deps = extract_dependencies(name, decl)
        dependencies[name].update(deps)
        # Attempt to identify relationships
        # Equivalence: look for statements like `lemma X : A = B` or `def X := Y`
        if 'def' in decl and ':=' in decl:
            rhs_match = re.search(r':=\s*(\S+)', decl)
            if rhs_match:
                rhs = rhs_match.group(1)
                if rhs in nodes:
                    equivalences.add((name, rhs))
        elif 'lemma' in decl or 'theorem' in decl:
            eq_match = re.search(r':\s*(\S+)\s*=\s*(\S+)', decl)
            if eq_match:
                lhs, rhs = eq_match.groups()
                if lhs in nodes and rhs in nodes:
                    equivalences.add((lhs, rhs))
        # You can add more sophisticated pattern matching to identify other relationships

# Now, print out the nodes and relationships
print("Nodes:")
for node in nodes:
    print(f"- {node}")

print("\nDependencies:")
for node, deps in dependencies.items():
    for dep in deps:
        print(f"Dep({node}, {dep})")

print("\nEquivalences:")
for eq in equivalences:
    print(f"Equ({eq[0]}, {eq[1]})")

# Since affiliations, opposites, synonyms, and properties require deeper semantic understanding,
# they are not extracted in this basic script. You can enhance the script by adding more
# sophisticated parsing and pattern matching to identify these relationships.

