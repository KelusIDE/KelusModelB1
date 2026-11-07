

import test from 'node:test';
import assert from 'node:assert/strict';
import { analysis, answer, keySentences, paragraph, pdfAttachment, prepareDocuments, sentences, wantsAnalysis, wantsParagraph } from '../../site/documents.js';
import { prepareAttachments, readAttachment } from '../../site/attachments.js';

const PAGES = [
  'Sea Otters of the Pacific\nSea otters live along the coasts of the northern Pacific Ocean. They spend almost their whole lives in the water, and they rarely come ashore.\nThe Otter Report  1',
  'Sea otters have the thickest fur of any animal, with up to a million hairs in each square inch. Unlike most marine mammals, sea otters have no blubber, so their fur keeps them warm.\nThe Otter Report  2',
  'Sea otters eat sea urchins, crabs and clams. An adult sea otter eats about a quarter of its weight in food every day. They use rocks as tools to crack open shells.\nThe Otter Report  3',
  'By eating sea urchins, sea otters protect kelp forests. Without otters, urchins can strip a kelp forest bare, so otters are called a keystone species.\nThe Otter Report  4',
];
const doc = pdfAttachment('otters.pdf', 1234, { title: 'Sea Otters', pages: PAGES, total: 4 });

test('sentences keep their page; repeated footers, broken hyphenation and glued headings are fixed', () => {
  const all = sentences(['A senten-\nce that was split across two lines in the PDF file.', ...PAGES]);
  assert.equal(all[0].text, 'A sentence that was split across two lines in the PDF file.');
  assert.ok(all.every(s => !/The Otter Report \d/.test(s.text)));
  assert.ok(all.some(s => s.page === 5 && /keystone species/.test(s.text)));
  const wrapped = 'Making honey\nForager bees collect nectar from flowers and carry it home in a special honey stomach that they\nhave. Back at the hive, they pass the nectar to house bees, who chew it and add enzymes to it.';
  assert.equal(sentences([wrapped])[0].text, 'Forager bees collect nectar from flowers and carry it home in a special honey stomach that they have.');
});

test('requests are told apart: analysis, a paragraph, or a question', () => {
  for (const prompt of ['', 'Analyze these files.', 'analyze this pdf', 'what is this document about?']) assert.ok(wantsAnalysis(prompt), prompt);
  for (const prompt of ['write a paragraph about otter food', 'summarize this', 'tl;dr', 'can you sum it up?']) assert.ok(wantsParagraph(prompt), prompt);
  for (const prompt of ['what do sea otters eat?', 'why are otters a keystone species?']) assert.ok(!wantsAnalysis(prompt) && !wantsParagraph(prompt), prompt);
});

test('the analysis gives size, topics and key sentences with their pages', () => {
  const report = analysis([doc]);
  assert.match(report, /^Sea Otters \(otters\.pdf\)\n4 pages · \d+ words · about 1 min to read/);
  assert.match(report, /Main topics: .*otters/i);
  assert.match(report, /• .+ \(p\. \d\)/);
  const empty = pdfAttachment('scan.pdf', 99, { title: null, pages: ['', ''], total: 2 });
  for (const text of [analysis([empty]), paragraph([empty], 'summarize this'), answer([empty], 'what?')]) assert.match(text, /scanned pages/);
});

test('a PDF from Chrome or Google Docs, with control characters for spaces, is read as text', () => {
  const page = 'Otter\u0001Facts\nSea\u0001otters\u0001hold\u0001hands\u0001while\u0001they\u0001sleep\u0001so\u0001they\u0001do\u0001not\u0001drift\u0001apart.';
  const saved = pdfAttachment('otters.pdf', 17081, { title: null, pages: [page], total: 1 });
  assert.equal(saved.words, 15);
  assert.doesNotMatch(saved.report, /scanned/);
  assert.match(answer([saved], 'why do sea otters hold hands?'), /drift apart/);
});

test('a paragraph about a subject is the sentences about it, in document order, with their pages', () => {
  const text = paragraph([doc], 'write a paragraph about what otters eat');
  assert.match(text, /^Sea otters eat sea urchins, crabs and clams\./);
  assert.match(text, /An adult sea otter eats about a quarter of its weight in food every day\. They use rocks as tools to crack open shells\. By eating sea urchins, sea otters protect kelp forests\.\n/);
  assert.ok(!/thickest fur|coasts|keystone/.test(text), 'sentences that only mention otters stay out');
  assert.match(text, /\n\nFrom otters\.pdf, pages 3, 4\. These are the PDF's own sentences/);
  assert.ok(!/Nothing in it matched/.test(text));
  assert.ok(!/Nothing in it matched/.test(paragraph([doc], 'summarize this')));
  assert.match(paragraph([doc], 'write a paragraph about volcanoes'), /Nothing in it matched your words, so this sums up its key points\./);
  assert.ok(!/^They /m.test(paragraph([doc], 'summarize this')), 'key sentences stand on their own');
});

test('with headings, a paragraph is the section about the subject', () => {
  const pages = ['Bees\nHoneybees live in colonies with one queen and thousands of workers that share the work of the hive.\nWorkers clean cells, feed young bees, guard the door and gather food from flowers nearby.\nMaking honey\nForager bees collect nectar from flowers and carry it home in a special honey stomach for later.\nBack at the hive, they pass the nectar to house bees, who chew it and add enzymes to it.\nThe bees fan the nectar with their wings until most of the water has gone and it turns thick.\nPeople and bees\nFarmers rent hives so that bees will pollinate their orchards and fields in the spring.'];
  const hive = pdfAttachment('bees.pdf', 1, { title: null, pages, total: 1 });
  const text = paragraph([hive], 'write a paragraph about how bees make honey');
  assert.match(text, /^Forager bees collect nectar .* Back at the hive, .* The bees fan the nectar .* turns thick\.\n\nFrom bees\.pdf, page 1\./);
});

test('a question is answered with the passage that matches it best, quoted with its page', () => {
  const warm = answer([doc], 'how do otters stay warm without blubber?');
  assert.match(warm, /^Page 2 of otters\.pdf:\n“.*no blubber, so their fur keeps them warm\.”/);
  assert.match(answer([doc], 'what do sea otters eat?'), /^Page 3 of otters\.pdf:\n“Sea otters eat sea urchins/);

  assert.match(answer([doc], 'do they use tools like rocks?'), /“An adult sea otter eats .* They use rocks as tools/);
  assert.match(answer([doc], 'what is the capital of France?'), /^I couldn't find that in otters\.pdf\. It is mostly about /);
  assert.match(answer([doc], 'who wrote this?'), /^otters\.pdf doesn't list an author/);
  const signed = pdfAttachment('otters.pdf', 1, { title: null, author: 'Ada Marsh', pages: PAGES, total: 4 });
  assert.equal(answer([signed], 'who wrote this pdf?'), 'otters.pdf lists Ada Marsh as its author.');
  assert.match(analysis([signed]), /^otters\.pdf\nBy Ada Marsh\n4 pages/);
});

test('key sentences skip near-duplicates', () => {
  const all = sentences(['Otters use rocks as tools to open shells. Otters use rocks as tools to open hard shells. Kelp forests need otters to stay healthy.']);
  assert.equal(keySentences(all, 3).length, 2);
});

test('attachments: PDFs are read by readPdf and answered locally; elsewhere they say where to read them', async () => {
  const bytes = new TextEncoder().encode('%PDF-1.7 fake');
  const file = { name: 'otters.pdf', size: bytes.length, arrayBuffer: async () => bytes.buffer };
  await assert.rejects(readAttachment(file), /kani\.kelus\.org/);
  const read = await readAttachment(file, { readPdf: async () => ({ title: null, pages: PAGES, total: 4 }) });
  assert.equal(read.kind, 'pdf');
  assert.match(read.report, /^4 pages · \d+ words/);
  const fake = { name: 'notes.pdf', size: 5, arrayBuffer: async () => new TextEncoder().encode('hello').buffer };
  await assert.rejects(readAttachment(fake, { readPdf: async () => ({}) }), /not a PDF/);
  const prepared = prepareAttachments('what do sea otters eat?', [read, read], null, null, 300);
  assert.deepEqual([prepared.helper, prepared.source], [true, 'document']);
  assert.match(prepared.text, /^Page 3 of otters\.pdf:/);
  assert.equal(prepareDocuments('summarize this', [read, read]).text.match(/From otters\.pdf/g).length, 1);
});
