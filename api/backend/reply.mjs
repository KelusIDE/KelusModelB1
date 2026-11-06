

import { CIDE_CALL, cideAnswer, maybeCide } from '../../site/cide.js';
import { PLACES_CALL, maybePlaces, placesAnswer } from '../../site/places.js';
import { SEARCH_CALL, maybeSearch, searchedAnswer } from '../../site/reading.js';
import { webAnswer, webRequest } from '../../site/web.js';
import { prepareReply } from '../../site/b1.js';

const pause = () => new Promise(setImmediate);







export async function* answer(model, messages, { maxTokens, sampling = {}, tools = true, coding = false, loadCide, cancelled = () => false, fetcher = fetch }) {
  if (coding) {
    yield { source: 'model' };
    yield* texts(cideAnswer(messages, { load: async () => model, maxTokens, sampling, cancelled, pause }));
    return;
  }
  const latest = messages.at(-1).content;
  const { retained, tool } = prepareReply(model, messages, maxTokens);
  if (tool) {
    yield { source: tool[1] };

    yield { text: tool[0].replace("this device's clock", "the Kani API server's clock") };
    return;
  }
  const asked = tools && webRequest(latest);
  if (asked) {
    yield { source: 'web' };
    if (asked.mode === 'search') yield* texts(searchedAnswer(asked.query, latest, { generate: generator(model, cancelled), webAnswer, fetcher }));
    else yield { text: (await webAnswer(asked, fetcher)).text };
    return;
  }

  let reply = '', held = true;
  for (const text of model.generate(retained, { ...sampling, maxTokens })) {
    if (cancelled()) return;
    reply += text;
    if (held && !maybeSearch(reply) && !maybePlaces(reply) && !maybeCide(reply)) { held = false; yield { source: 'model' }; yield { text: reply }; }
    else if (!held) yield { text };
    await pause();
  }
  if (!held) return;
  const call = reply.trim();
  if (call === CIDE_CALL && loadCide) {
    yield { source: 'cide' };
    yield* texts(cideAnswer(messages, { load: loadCide, maxTokens, sampling, cancelled, pause }));
    return;
  }
  const place = tools && call.match(PLACES_CALL), search = tools && call.match(SEARCH_CALL);
  if (place) {

    yield { source: 'places' };
    yield* texts(placesAnswer(place, { locate: async () => null, fetcher }));
  } else if (search) {
    yield { source: 'web' };
    yield* texts(searchedAnswer(search[1], latest, { generate: generator(model, cancelled), webAnswer, fetcher }));
  } else {
    yield { source: 'model' };
    if (reply) yield { text: reply };
  }
}

async function* texts(pieces) {
  for await (const text of pieces) yield { text };
}

function generator(model, cancelled) {
  return async function* (messages) {
    for (const text of model.generate(messages, { maxTokens: 200, temperature: 0 })) {
      if (cancelled()) return;
      yield text;
      await pause();
    }
  };
}
