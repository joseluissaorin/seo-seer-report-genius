
import { Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-4 text-center text-gray-600 border-t">
      <p className="flex items-center justify-center gap-1">
        made with <Heart className="h-4 w-4 text-red-500 fill-red-500" /> by saorin
      </p>
      <p>© 2025</p>
    </footer>
  );
};

export default Footer;
