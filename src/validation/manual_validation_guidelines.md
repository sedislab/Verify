# Manual Validation Guidelines: LTL, ITL, and NL Triplets

## 1. Our Goal

Hi there! Thanks for helping us validate the VERIFY dataset. Our main goal is to ensure that the Natural Language (NL) descriptions accurately and clearly represent the formal logic expressed in the Linear Temporal Logic (LTL) formulas and their Intermediate Technical Language (ITL) counterparts. We need to make sure these translations are not just understandable, but truly equivalent in meaning, especially concerning the timing and sequence of events.

## 2. What You'll Be Looking At

For each item you validate, you'll see a set of information:

* **LTL Formula**: The formal specification using standard LTL operators (like G for Always, F for Eventually, X for Next, U for Until, R for Release, W for Weak Until, and logical connectives like !, &, |, ->, <->). This is the ground truth for the logical meaning.
    * *Example LTL*: `G(request -> F(response))`
* **ITL Representation**: A more structured, English-like version of the LTL formula. It's meant to be a bridge between the formal LTL and the more free-form NL.
    * *Example ITL (for above LTL)*: `Always, if request, then Eventually, response`
* **Domain**: The specific application area the NL description is set in (e.g., "Home Automation", "Robotics").
* **Activity Context**: Definitions for the atomic propositions (like 'p', 'q', 'request', 'response') used in the LTL/ITL, specific to the given domain. This tells you what the basic terms mean.
    * *Example Activity (for "Robotics" domain)*: `request = robot arm receives a pick-up command; response = robot arm successfully grasps the object`
* **Natural Language (NL) Translation**: The human-readable sentence or sentences that are supposed to convey the meaning of the LTL/ITL within the given domain and activity context. **This is what you'll be primarily judging.**
    * *Example NL (for above LTL, ITL, Activity)*: "In our robotics system, it must always be the case that if the robot arm receives a pick-up command, then it will eventually successfully grasp the object."

## 3. Core Validation Criteria

When you're looking at each NL translation, I want you to focus on these three main aspects:

### A. Semantic Equivalence (Is the meaning the same?)

This is the **most critical** part. Does the NL translation mean *exactly* the same thing as the LTL formula?
* **Temporal Relationships**: Are operators like "Always" (G), "Eventually" (F), "Next" (X), "Until" (U), "Release" (R), and "Weak Until" (W) correctly conveyed? For instance, "eventually" is different from "in the next state."
* **Logical Structure**: Are logical connectives like AND, OR, NOT, IMPLIES (if...then...), and IFF (if and only if) accurately represented?
* **No Added/Lost Meaning**: Does the NL introduce new conditions or omit any conditions present in the LTL?

### B. Contextual Relevance (Does it make sense in the domain?)

* **Activity Consistency**: Is the NL translation consistent with the provided "Activity Context"? Do the actions and states described in the NL match what the propositions are defined as?
* **Domain Plausibility**: Does the scenario described by the NL and the activity definitions seem reasonable for the given "Domain"? (e.g., a financial transaction scenario shouldn't talk about robot arms unless the activity context makes a very clear, albeit unusual, link).

### C. Linguistic Quality (Is it well-written?)

* **Fluency**: Does the NL translation read naturally and smoothly?
* **Grammar & Clarity**: Is it grammatically correct and easy to understand? Is it free of jargon that wouldn't be clear to someone familiar with the domain but not necessarily LTL?
* **Conciseness**: Is it to the point, or is it overly wordy? (While still needing to be precise).

## 4. My Suggested Validation Process

Here's how I'd approach each item:

1.  **Understand the LTL First**: Look at the LTL formula. Try to understand what property it's specifying. Pay close attention to the temporal operators and their scope.
2.  **Review the ITL**: See how the ITL represents the LTL. This can help bridge your understanding to the NL.
3.  **Read the Domain and Activity Context**: This is crucial. Understand what 'p', 'q', etc., mean in this specific scenario.
4.  **Carefully Read the NL Translation**.
5.  **Compare NL to LTL (Semantic Equivalence Check)**:
    * Go operator by operator if needed. Does "Always" in the LTL truly mean "always" in the NL?
    * Does "p until q" in LTL mean the same as how the NL describes the relationship between the activities for 'p' and 'q'?
6.  **Check Contextual Relevance**: Does the story told by the NL make sense given the domain and the definitions in the activity context?
7.  **Assess Linguistic Quality**: Read the NL aloud. Does it sound like something a person would naturally write or say?

## 5. Key Areas to Focus On (With Examples)

Let's dive into some specifics.

### Temporal Operators: The Heart of LTL

* **G (Globally/Always)**:
    * *LTL*: `G(light_on)`
    * *Activity*: `light_on = The security light is activated.`
    * *Good NL*: "The security light must always be activated." or "It is always the case that the security light is activated."
    * *Bad NL*: "The security light is usually activated." (Usually is not Always) or "The security light will be activated." (Future, but not necessarily *always* from now on).
* **F (Finally/Eventually)**:
    * *LTL*: `F(payment_processed)`
    * *Activity*: `payment_processed = The customer's payment is successfully processed.`
    * *Good NL*: "The customer's payment will eventually be successfully processed." or "At some point in the future, the customer's payment will be successfully processed."
    * *Bad NL*: "The customer's payment is processed in the next step." (This is X, not F) or "The customer's payment might be processed." (Might is not will eventually).
* **X (Next)**:
    * *LTL*: `X(door_locked)`
    * *Activity*: `door_locked = The vault door is in a locked state.`
    * *Good NL*: "In the immediately following state, the vault door must be locked." or "The vault door will be locked in the next time step."
    * *Bad NL*: "The vault door will be locked soon." (Soon is F, not X).
* **U (Until)**: `p U q` means 'p' must be true at least until 'q' becomes true. 'q' *must* eventually become true.
    * *LTL*: `access_attempt U login_success`
    * *Activity*: `access_attempt = The system is processing an access attempt; login_success = The user has successfully logged in.`
    * *Good NL*: "The system will keep processing the access attempt until the user successfully logs in." (Implies login_success must happen).
    * *Bad NL*: "The system processes an access attempt, and then the user might log in." (Doesn't capture 'until' or the guarantee of `login_success`).
* **R (Release)**: `p R q` means 'q' must be true until and including the point where 'p' becomes true. If 'p' never becomes true, 'q' must remain true forever. It's like 'q' is the default, and 'p' can release it.
    * *LTL*: `error_flag R system_stable`
    * *Activity*: `error_flag = An error flag is raised; system_stable = The system is in a stable operating condition.`
    * *Good NL*: "The system must remain in a stable operating condition until an error flag is raised, and it must also be stable at the moment the error flag is raised." or "The system is stable, and remains stable, up to and including the point where an error flag is raised; if no error flag is ever raised, the system remains stable indefinitely."
    * *Bad NL*: "If an error flag is raised, the system is stable." (Misses the continuous nature of `system_stable`).
* **W (Weak Until)**: `p W q` is like `p U q`, but 'q' is *not* guaranteed to become true. 'p' can hold true indefinitely.
    * *LTL*: `heater_on W temp_reached`
    * *Activity*: `heater_on = The heater is active; temp_reached = The target temperature is achieved.`
    * *Good NL*: "The heater will remain active until the target temperature is achieved, or it will remain active indefinitely if the target temperature is never achieved."
    * *Bad NL*: "The heater is on until the temperature is reached." (This sounds like strong Until, implying `temp_reached` must happen).

### Logical Connectives

* **! (Not)**: Ensure negation applies to the correct part of the statement.
    * *LTL*: `G(!(door_open & alarm_active))`
    * *Good NL*: "It is always the case that it's not true that both the door is open and the alarm is active." (Better: "It is never the case that the door is open and the alarm is active simultaneously.")
    * *Bad NL*: "The door is always closed and the alarm is always inactive." (This is `G(!door_open) & G(!alarm_active)`, which is different).
* **& (And)**: All conditions must hold.
* **| (Or)**: At least one condition must hold.
* **-> (Implies / If...then...)**: `p -> q` means "if p is true, then q must also be true." It doesn't say anything about q if p is false.
    * *LTL*: `low_battery -> activate_power_save`
    * *Good NL*: "If the battery is low, then power saving mode must be activated."
    * *Bad NL*: "Power saving mode is activated only if the battery is low." (This is `activate_power_save -> low_battery`).
* **<-> (Iff / If and only if)**: `p <-> q` means p and q always have the same truth value (both true or both false).

### Atomic Propositions and Activity Context

* Does the NL use terms that accurately reflect the definitions in the "Activity Context"?
* If `p = "the light is red"`, the NL shouldn't say "the light is green" when referring to 'p'.

### Domain Specificity

* Does the language used in the NL feel appropriate for the specified "Domain"? For example, "Financial/Transaction Systems" will use different terminology than "Robotics".

### Fluency and Clarity

* Is the NL awkward or convoluted?
* Could it be misunderstood easily?

## 6. Common Pitfalls to Watch For

* **Weakening Strong Conditions**: "Eventually" (F) becoming "possibly" or "maybe." "Always" (G) becoming "usually" or "often."
* **Strengthening Weak Conditions**: "Weak Until" (W) being described as a strong "Until" (U).
* **Misplaced Negations**: `!G(p)` (it's not always p / eventually not p) is different from `G(!p)` (it's always not p).
* **Incorrect Scope**: In `G(p -> F(q))`, the "Always" applies to the entire "if p then eventually q" implication. An incorrect NL might say "If p is always true, then q is eventually true."
* **Overly Literal Translations**: Sometimes the ITL is very direct. The NL should be more natural while keeping the exact meaning. "Always, if p, then q" is fine for ITL, but NL might be "Whenever p occurs, q must also occur."
* **Ignoring the "Activity Context"**: Translating `G(p)` as "Always p" without using the definition of 'p' from the activity context.

## 7. Recording Your Judgment

When you've evaluated an item, I'll need you to record:

1.  **Is it Correct?**: A simple `Yes` or `No`.
    * `Yes`: If it perfectly meets semantic equivalence, is contextually relevant, and has good linguistic quality. Minor stylistic preferences don't make it "No" if the meaning is spot on.
    * `No`: If there's any issue with semantic equivalence, or significant problems with context or language. **When in doubt, lean towards "No" if the meaning isn't perfectly preserved.**
2.  **Score (0-10)**: (If we're using a scoring system)
    * 10: Perfect.
    * 8-9: Minor linguistic awkwardness, but semantically perfect.
    * 5-7: Semantically mostly correct but with some ambiguity or minor error in temporal/logical meaning, or significant fluency issues.
    * 0-4: Semantically incorrect or misleading.
3.  **Issues (Brief Description)**: If "No" or score is low, please briefly note why.
    * *Examples*: "Mistakes F for X", "NL implies 'p' is optional when LTL says it's required", "Activity context not used", "Awkward phrasing".

## 8. Guiding Principles

* **Precision is Key**: Semantic equivalence is paramount.
* **Think Temporally**: The LTL operators define behavior *over time*. Ensure the NL captures this.
* **Consider the Audience**: The NL should be clear to someone in that domain who might not be an LTL expert.
* **When in Doubt, Ask!**: If you're unsure about an LTL construct or how to judge an NL, please reach out.

Thanks again for your help! Your careful validation is essential to the quality of the VERIFY dataset.
