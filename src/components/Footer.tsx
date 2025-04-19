
import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-background text-foreground text-center py-4 mt-8 border-t">
      <div className="container mx-auto px-4">
        <p className="text-sm">
          Made with ❤️ by Saorin • © {new Date().getFullYear()}
        </p>
      </div>
    </footer>
  );
};

export default Footer;
