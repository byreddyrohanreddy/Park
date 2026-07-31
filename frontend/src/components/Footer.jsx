import { FaFacebook, FaInstagram, FaTwitter } from "react-icons/fa";
import { MdEmail } from "react-icons/md";

function Footer() {
  return (
    <footer className="bg-slate-900 text-white mt-20">
      <div className="max-w-7xl mx-auto px-6 py-10 grid md:grid-cols-3 gap-8">
        <div>
          <h2 className="text-2xl font-bold">VoiceCare PD</h2>
          <p className="text-gray-400 mt-3">
            Parkinson's Disease Voice Analysis Platform
          </p>
        </div>

        <div>
          <h3 className="font-semibold">Contact</h3>
          <p className="text-gray-400 mt-3">support@voicecare.com</p>
          <p className="text-gray-400">+91 9876543210</p>
        </div>

        <div>
          <h3 className="font-semibold">Follow Us</h3>

          <div className="flex gap-4 mt-4 text-xl">
            <FaFacebook />
            <FaInstagram />
            <FaTwitter />
            <MdEmail />
          </div>
        </div>
      </div>

      <p className="text-center border-t border-gray-700 py-4 text-gray-400">
        © 2026 VoiceCare PD. All rights reserved.
      </p>
    </footer>
  );
}

export default Footer;