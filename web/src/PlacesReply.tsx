import { useState } from 'react';
import type { Place, PlacesResult } from './engine';

const sentence = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
const names = (list: string[]) => list.length < 3 ? list.join(' and ') : `${list.slice(0, -1).join(', ')} and ${list.at(-1)}`;
const OUT = { target: '_blank', rel: 'noopener noreferrer' } as const;


export function PlacesReply({ result }: { result: PlacesResult }) {
  const { what, where, places, source } = result;
  const area = where.toLowerCase() === 'here' ? 'near you' : `in ${where}`;
  const known = places.filter(place => place.wiki).map(place => place.name);
  const credits = [`Places from ${source || 'a map service'}`, known.length && 'descriptions and photos from Wikipedia',
    places.some(place => place.image?.includes('/map?')) && 'maps © Mapbox © OpenStreetMap'].filter(Boolean).join(' · ');
  return (
    <div className="places reply">
      <p className="places-intro">
        I found {places.length === 1 ? 'one place' : `${places.length} places`} for {what} {area}.
        {known.length > 0 && ` ${names(known.slice(0, 2))} ${known.length === 1 ? 'is' : 'are'} the best known.`}
      </p>
      <h3 className="places-title">{sentence(what)} {area}</h3>
      <ol className="place-list">
        {places.map((place, i) => <PlaceCard key={`${place.name}-${i}`} place={place} number={i + 1} />)}
      </ol>
      <p className="places-credit">{credits}</p>
    </div>
  );
}

function PlaceCard({ place, number }: { place: Place; number: number }) {
  const [broken, setBroken] = useState(false);
  const map = place.image?.includes('/map?');
  const facts = [place.category && sentence(place.category), place.area,
    place.rating != null && `★ ${place.rating.toFixed(1)}${place.ratings ? ` (${place.ratings.toLocaleString('en-US')})` : ''}`,
    place.price, place.open === true ? 'Open now' : place.open === false ? 'Closed now' : null].filter(Boolean);
  return (
    <li className="place">
      <a className="place-picture" href={place.wiki || place.url} {...OUT} tabIndex={-1} aria-hidden="true">
        {place.image && !broken
          ? <img src={place.image} alt="" loading="lazy" referrerPolicy="no-referrer" className={map ? 'map' : 'photo'} onError={() => setBroken(true)} />
          : <svg viewBox="0 0 24 24" className="place-pin"><path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5z" /></svg>}
      </a>
      <div className="place-body">
        <a className="place-name" href={place.url} {...OUT}>{number}. {place.name}</a>
        {facts.length > 0 && <div className="place-facts">{facts.join(' · ')}</div>}
        {(place.about || place.address) && (
          <p className="place-about">
            {place.about || place.address}
            {place.wiki && <> <a className="place-chip" href={place.wiki} {...OUT}>Wikipedia</a></>}
          </p>
        )}
        <div className="place-links">
          <a href={place.url} {...OUT}>Directions <span aria-hidden="true">↗</span></a>
          {place.website && <a href={place.website} {...OUT}>Website <span aria-hidden="true">↗</span></a>}
        </div>
      </div>
    </li>
  );
}
