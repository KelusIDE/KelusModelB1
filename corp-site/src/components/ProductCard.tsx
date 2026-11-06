import { motion } from "motion/react";
import type { Product } from "../products";
import { Arrow } from "./Arrow";

export function ProductCard({
  product,
  index,
}: {
  product: Product;
  index: number;
}) {
  return (
    <motion.article
      className={`product-card ${product.featured ? "featured" : "compact"} product-${product.id}`}
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.45, delay: (index % 2) * 0.05 }}
    >
      <div className="card-top">
        {"src" in product.icon ? (
          <img
            className="product-icon"
            src={product.icon.src}
            alt=""
            width={48}
            height={48}
          />
        ) : (
          <span className="product-icon glyph" aria-hidden="true">
            {product.icon.glyph}
          </span>
        )}
        <span className={`status ${product.status}`}>
          <i />
          {product.statusLabel}
        </span>
      </div>
      <p className="kind">{product.kind}</p>
      <h3>{product.name}</h3>
      <p className="product-description">{product.description}</p>
      <ul className="chips" aria-label="Details">
        {product.chips.map((chip) => (
          <li key={chip}>{chip}</li>
        ))}
      </ul>
      <div className="product-links">
        {product.links.map((link, i) => (
          <a
            key={link.href}
            className={
              product.featured && i === 0 ? "button card-button" : "text-link"
            }
            href={link.href}
          >
            {link.label}
            <Arrow diagonal />
          </a>
        ))}
      </div>
      {product.note && <p className="fine">{product.note}</p>}
      {product.featured && (
        <span className="card-watermark" aria-hidden="true">
          {product.id === "ide" ? "{ }" : "hello."}
        </span>
      )}
    </motion.article>
  );
}
