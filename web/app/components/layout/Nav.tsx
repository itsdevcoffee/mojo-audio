import { NavLink } from "@remix-run/react";

const links = [
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/analyzer", label: "Analyzer" },
  { to: "/docs", label: "Docs" },
];

export function Nav() {
  return (
    <nav className="nav">
      <NavLink to="/" className="nav-brand gradient-text">
        mojo-audio
      </NavLink>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          className={({ isActive }) =>
            `nav-link ${isActive ? "nav-link--active" : ""}`
          }
        >
          {link.label}
        </NavLink>
      ))}
      <a
        href="https://github.com/dev-coffee/mojo-audio"
        target="_blank"
        rel="noopener noreferrer"
        className="nav-link nav-link--github"
      >
        GitHub ↗
      </a>
    </nav>
  );
}
