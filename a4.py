
# PR4: Rule-Based Expert System

rules = [
    (['fever', 'cough'], 'flu'),
    (['flu'], 'rest_needed'),
    (['fever', 'body_pain'], 'infection')
]

facts = {'fever', 'cough', 'body_pain'}


def forward_chaining(facts):
    facts = facts.copy()

    while True:
        new_fact = False

        for conditions, result in rules:
            if all(c in facts for c in conditions) and result not in facts:
                facts.add(result)
                print("Derived:", result)
                new_fact = True

        if not new_fact:
            break

    return facts


def backward_chaining(goal, facts):
    if goal in facts:
        return True

    for conditions, result in rules:
        if result == goal:
            if all(backward_chaining(c, facts) for c in conditions):
                return True

    return False


print("Forward Chaining:")
all_facts = forward_chaining(facts)
print("All facts:", all_facts)

goal = 'rest_needed'
print("\nBackward Chaining:")
print(goal, ":", backward_chaining(goal, facts))
