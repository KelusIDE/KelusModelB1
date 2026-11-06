// Fixed replies for people who may be in danger, shared by every app. These never go to the model: Kani is too small to
// be trusted here (Otter 4.11 answered "i want to die" with "Fun! Do you know what you'd like to do next?").
// kelus/local_tools.py holds the same patterns and replies; tests/test_web.py checks that the two agree.
// No lookbehind in these patterns: Safari before 16.4 rejects the whole file.

export const CRISIS_REPLY = "I'm really sorry you're in this much pain. You deserve support right now, and I'm only a small AI, "
  + 'so please reach out to a person. In the US, call or text 988 for the Suicide and Crisis Lifeline. '
  + 'Elsewhere, call your local emergency number or a crisis line, or tell someone you trust right away.';
export const DANGER_REPLY = 'Please get help right now. Call your local emergency number (911 in the US), or ask someone near you to call. '
  + 'If you or someone else took something, US Poison Help is 1-800-222-1222, day and night. Please don\'t wait to see how you feel, '
  + "and don't stay alone. I'm only a small AI, so a person needs to help with this.";
export const OTHER_PERSON_REPLY = "That sounds really scary, and it's good that you're taking it seriously. Stay with them if you can, and "
  + "don't keep it a secret, even if they ask you to. In the US, you can call or text 988 together, or on your own to ask how to help. "
  + 'If they are in immediate danger, call your local emergency number now, and tell an adult or someone you trust.';
export const LOSS_REPLY = "I'm so sorry for your loss. Losing someone this way is incredibly painful, and whatever you're feeling is okay. "
  + "Please don't carry it alone, and talk to someone you trust. In the US, 988 is there day and night for anyone who is struggling, "
  + 'including after losing someone to suicide.';
export const CHECK_IN_REPLY = "That sounds really heavy, and I'm glad you told me. You don't have to carry it alone. If you're thinking about "
  + 'ending your life or hurting yourself, please reach out to a person now: in the US, call or text 988. Elsewhere, call your local '
  + "emergency number or a crisis line. If it's something else, what's been going on?";
export const STILL_HERE_REPLY = "I'm still here, and I'm glad you're talking. I'm only a small AI and can't keep you safe the way a person can, "
  + 'so please call or text 988 in the US, or your local crisis line or emergency number elsewhere, or tell someone near you what you '
  + 'told me. Are you safe right now?';

// Removed before matching, so that "i cut myself shaving and i want to die" is still caught.
const HARMLESS = new RegExp(String.raw`\bsuicide (?:squad|doors?|burn|king|forest)\b|\bkill(?:ing)? my ?self laughing\b`
  + String.raw`|\b(?:hurt(?:s|ing)?|cut(?:s|ting)?|harm(?:s|ing)?) (?:my ?self|himself|herself|themsel(?:f|ves))(?: \w+)? (?:while|when|playing|shaving|cooking|by accident|accidentally|at (?:the gym|work|practice)|on (?:a|an|the|my))\b`, 'g');
const DANGER = new RegExp(String.raw`\b(?:took|taken|take|swallowed|ate)\b(?: \w+)? (?:too many|too much|a bunch of|a lot of|a whole bottle of|the whole bottle of|handfuls? of|all (?:of )?(?:my|the))\b.{0,30}\b(?:pills?|meds|medication|medicine|tablets?|painkillers|tylenol)\b`
  + String.raw`|\boverdos(?:ed|ing)\b|^(?:no,? )?i'?m not safe\b`);
const SELF = new RegExp(String.raw`\b(?:kill(?:ing)?|hurt(?:ing)?|harm(?:ing)?|cut(?:ting)?|unalive|hang(?:ing)?|shoot(?:ing)?|drown(?:ing)?|poison(?:ing)?|stab(?:bing)?) ?my ?self\b`
  + String.raw`|\bslit(?:ting)? my wrists?\b|\b(?:want|wanna|wanting|ready)(?: to)? (?:just )?die\b|\bwish i (?:could|would) (?:just )?die\b`
  + String.raw`|\bend(?:ing)? (?:my (?:own )?life|it all|my ?self)\b|\b(?:want to|wanna|going to|gonna) end it(?: tonight| today| now)?$|\btake my (?:own )?life\b`
  + String.raw`|\b(?:don'?t|do not) want to (?:live|be alive|exist|be here|wake up)\b|\bstop existing\b|\bgo on living\b`
  + String.raw`|(?:\bnot|n'?t) worth living\b|\bwish i (?:was|were) dead\b|\bwish i (?:wasn'?t|weren'?t) alive\b|\bwish i(?:'d| had| was| were)? never (?:been )?born\b|\bbetter off (?:dead|without me)\b`
  + String.raw`|\b(?:wish|want|hope)\b.{0,40}\b(?:never|not|don'?t|wouldn'?t) wake up(?: again| tomorrow| anymore)?$`
  + String.raw`|\b(?:no|see a|see any|find a|find any) (?:reason|point) (?:to|in) (?:live|living|keep going|go on|going on)\b|\bnothing to live for\b`
  + String.raw`|\b(?:tired|sick) of (?:living|being alive)\b|\bwould care if i died\b|\b(?:nobody|no one) would (?:care|notice|miss me)\b.{0,12}\bif i (?:died|was gone|were gone|disappeared)\b`
  + String.raw`|\bdisappear forever\b|(?:^|\b(?:i|i'?m|gonna|wanna|to|will|should|just|might)\s)kms\b`
  + String.raw`|\b(?:rather|better) be dead\b|\b(?:had|have) enough of (?:living|life|being alive)\b|\bcan'?t (?:keep|go on|continue) living\b|\bthink(?:ing)? (?:about|of) (?:not being (?:alive|here)|not existing|being dead|dying)\b|\b(?:easier|better)\b.{0,30}\bif i (?:wasn'?t|weren'?t|was not|were not) (?:here|around|alive)\b|\bif i (?:was|were) (?:gone|dead)\b.{0,30}\b(?:easier|better)\b|\b(?:pills?|rope|gun)\b.{0,30}\bend (?:it|things|my)\b`
  + String.raw`|\b(?:(?:want|wanna|going|gonna|might|will)(?: to)?) jump off (?:a|the|this) (?:bridge|building|roof|cliff|balcony)\b`
  + String.raw`|\b(?:i'?m|i am|feel(?:ing)?|been|have|having|getting)\b.{0,24}\bsuicid`);
const LOSS = new RegExp(String.raw`\bmy\b.{0,40}\b(?:killed (?:him|her|them)sel(?:f|ves)|committed suicide|died by suicide|took (?:his|her|their) (?:own )?life)\b|\blost\b.{0,30}\bto suicide\b`);
const OTHER_WHO = new RegExp(String.raw`\b(?:friend|bestie|bff|sister|brother|sibling|mom|mum|mother|dad|father|parent|son|daughter|kid|child|boyfriend|bf|girlfriend|gf|wife|husband|partner|cousin|aunt|uncle|classmate|roommate|coworker|colleague|student|neighbor|teammate|grandma|grandpa|someone|somebody|he|she)\b`);
const OTHER_ACT = new RegExp(String.raw`\b(?:kill(?:s|ing)?|hurt(?:s|ing)?|harm(?:s|ing)?|cut(?:s|ting)?|unalive(?:s)?) (?:him|her|them)sel(?:f|ves)\b|\bend(?:s|ing)? (?:his|her|their) (?:own )?life\b|\bself[- ]?harm|\bsuicid|\boverdos|\bwants? to die\b|\bwanna die\b`);
const GENERAL = new RegExp(String.raw`\bsuicid|\bself[- ]?harm|\boverdos`);
const CHECK_IN = new RegExp(String.raw`\b(?:what'?s|what is) the point(?: of (?:living|life|anything|it all|going on|being alive|trying))?(?: any ?more)?$`
  + String.raw`|\b(?:don'?t|do not|can'?t) see the point(?: of (?:anything|living|life|going on|trying))?(?: any ?more)?$`
  + String.raw`|\bi can'?t (?:do this|do it|take (?:it|this)|go on|keep going|cope|handle (?:it|this|life))(?: any ?more| any longer)$|\bi can'?t go on$`
  + String.raw`|\bi hate (?:my life|myself|being alive)$|\bmy life is (?:over|pointless|worthless|meaningless|ruined)$|\blife is (?:pointless|meaningless)$`
  + String.raw`|\bi'?m (?:just )?(?:a burden|worthless)(?: to (?:everyone|everybody|my family|my friends|anyone))?$`
  + String.raw`|\bi (?:should|want to|wanna|wish i could) (?:just )?disappear$|\bwhy (?:should|would|do) i (?:even )?(?:keep going|go on|bother living|stay alive|live)\b`);
const NEW_SUBJECT = /^(?:(?:hi|hey|ok|okay|thanks|thank you|anyway)[,. ]+)?(?:what|who|when|where|which|how|why|can you|could you|write|make|build|create|draw|tell me|explain|define|translate|search|find|show|give|calculate|solve|code)\b/;

export const normalized = text => text.toLowerCase().replaceAll('’', "'").split(/\s+/).filter(Boolean).join(' ').replace(/[?! .]+$/, '');

/** The fixed reply for a message (already lower case, as `normalized` gives), or null. */
export function supportReply(query) {
  const text = query.replace(HARMLESS, ' ').replace(/\s+/g, ' ').trim();
  if (DANGER.test(text)) return DANGER_REPLY;
  if (SELF.test(text)) return CRISIS_REPLY;
  if (LOSS.test(text)) return LOSS_REPLY;
  if (OTHER_WHO.test(text) && OTHER_ACT.test(text)) return OTHER_PERSON_REPLY;
  if (GENERAL.test(text)) return CRISIS_REPLY;
  return CHECK_IN.test(text) ? CHECK_IN_REPLY : null;
}

/** The next message after a fixed reply to someone in danger stays with a person's help, unless it plainly moves on.
 *  Only one turn: after that the model answers, or "yes I'm safe" would get the same text forever. */
export function supportFollowUp(messages) {
  const earlier = messages.at(-2);
  if (earlier?.role !== 'assistant' || ![CRISIS_REPLY, DANGER_REPLY].includes(earlier.content)) return null;
  const query = normalized(messages.at(-1).content);
  const movedOn = NEW_SUBJECT.test(query) && query.split(' ').length >= 3 && !/\b(?:i|i'?m|me|my|myself)\b/.test(query);
  return movedOn ? null : STILL_HERE_REPLY;
}
