
import { Heart, Copyright } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-6 text-center text-muted-foreground border-t bg-background/90 backdrop-blur-sm animate-fade-in">
      <div className="container mx-auto px-4">
        <p className="flex items-center justify-center gap-2 text-sm">
          made with 
          <Heart className="h-4 w-4 text-red-500 fill-red-500 animate-pulse" /> 
          by <span className="font-semibold text-primary hover:text-primary/80 transition-colors">saorin</span>
        </p>
        <div className="flex items-center justify-center gap-1 mt-2 text-xs">
          <Copyright className="h-3 w-3" />
          <span>{new Date().getFullYear()}</span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
