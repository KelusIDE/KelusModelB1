


export const RECIPE_QUERY = /\brecipes?\b/i;

const UNIT = '(?:cups?|c\\.|tablespoons?|tbsps?|tbs|teaspoons?|tsps?|grams?|g|kg|kilograms?|ml|millilit(?:er|re)s?|l|lit(?:er|re)s?|'
  + 'oz|ounces?|pounds?|lbs?|pinch(?:es)?|dash(?:es)?|cloves?|sticks?|cans?|slices?|large|medium|small|whole|handfuls?|eggs?|'
  + 'el|tl|cucharadas?|cucharaditas?|tazas?|colher(?:es)?|x[ií]caras?|cuill[eè]res?|tasses?|eetlepels?|theelepels?|kopjes?|stuks?|'
  + 'eier|prisen?|stück|bund|becher|päckchen|huevos?|pizcas?|œufs?|oeufs?|pincées?|ovos?|pitadas?|eieren|snufjes?|uova|uovo|pizzico|cucchiai(?:ni|no)?)';

const AMOUNT = '(?:\\d+(?:[.,]\\d+)?(?:\\s*[½¼¾⅓⅔⅛⅜⅝⅞]|\\s+\\d+[/⁄]\\d+)?|\\d+[/⁄]\\d+|[½¼¾⅓⅔⅛⅜⅝⅞])';
const BULLET = /\s*[▢☐□■•◦▪●]\s*/g;

const ITEM = new RegExp(`(?<!\\d)\\s(?=${AMOUNT}(?:\\s*(?:-|–|to)\\s*${AMOUNT})?\\s*${UNIT}\\b)`, 'gi');
const GROUP = /\s(?=(?:dry|wet|for the [a-z]+|other)\s+ingredients?\s*:)/gi;

const INGREDIENT_WORDS = 'ingredients|zutaten|ingrédients|ingredientes|ingrediënten|ingredienti';
const STEP_WORDS = 'method|instructions|directions|preparation|steps|how to make (?:it|them)|zubereitung|anleitung|préparation|étapes|'
  + 'preparación|elaboración|modo de preparo|preparo|bereiding|bereidingswijze|procedimento|preparazione';
const INGREDIENTS = new RegExp(`(?<![\\p{L}])(?:${INGREDIENT_WORDS})(?![\\p{L}])\\s*:?`, 'iu');
const STEPS = new RegExp(`(?<![\\p{L}])(?:${STEP_WORDS})(?![\\p{L}])\\s*:?`, 'iu');

const HEADING = new RegExp(`(?<![\\p{L}])(${INGREDIENT_WORDS}|${STEP_WORDS})(?=[\\p{Lu}0-9])`, 'giu');

const capitalized = words => words.split('|').map(w => w[0].toUpperCase() + w.slice(1)).join('|');
const GLUED = new RegExp(`(\\p{Ll})(?=(?:${capitalized(INGREDIENT_WORDS)}|${capitalized(STEP_WORDS)}))`, 'gu');


export function recipeOf(text) {
  const spaced = String(text || '').replace(/\s+/g, ' ').replace(GLUED, '$1 ').replace(HEADING, '$1 ').replace(/:(?=\S)/g, ': ');
  const start = spaced.search(INGREDIENTS);
  if (start < 0) return null;
  const body = ' ' + spaced.slice(start).replace(INGREDIENTS, '');
  const split = body.search(STEPS);
  const ingredients = (split > 0 ? body.slice(0, split) : body).split(GROUP).flatMap(part => part.split(BULLET)).flatMap(part => (' ' + part).split(ITEM))
    .map(item => item.trim().replace(/[,;]$/, '')).filter(item => item.length > 1 && item.length <= 140);
  if (ingredients.filter(item => new RegExp(`^${AMOUNT}`, 'i').test(item)).length < 3) return null;
  const sentences = split > 0 ? body.slice(split).replace(STEPS, '').trim().split(/(?<=[.!?])\s+(?=[A-Z0-9])/) : [];
  const complete = sentences.length > 0 && /[.!?]$/.test(sentences.at(-1).trim());
  const steps = sentences.map(s => s.trim()).filter(s => s.length > 3 && (complete || s !== sentences.at(-1).trim())).slice(0, 15);
  return { ingredients: ingredients.slice(0, 25), steps, cut: !complete || sentences.length > 15 || ingredients.length > 25 };
}

const host = url => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return 'the web'; } };


export function recipeAnswer(dish, sources) {
  const found = sources.map((source, index) => ({ source, index, recipe: recipeOf(source.text) })).filter(f => f.recipe)
    .sort((a, b) => b.recipe.ingredients.length + 2 * b.recipe.steps.length - a.recipe.ingredients.length - 2 * a.recipe.steps.length)[0];
  if (!found) return null;
  const { ingredients, steps, cut } = found.recipe, n = found.index + 1;
  const name = dish.replace(RECIPE_QUERY, '').trim() || 'this dish';
  return [`Here's a recipe for ${name} from ${host(found.source.url)} [${n}]:`, '',
    'Ingredients:', ...ingredients.map(item => item.endsWith(':') ? item : `- ${item}`), '',
    ...(steps.length ? ['Steps:', ...steps.map((step, i) => `${i + 1}. ${step}`)] : [`The steps are on the page [${n}].`]),
    ...(cut ? ['', `This is the start of the recipe; the full one is at the link [${n}].`] : [])].join('\n');
}
