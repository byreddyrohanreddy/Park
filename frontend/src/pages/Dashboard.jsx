import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Mic,
  Upload,
  History,
  FileAudio,
  User
} from "lucide-react";
import { Link } from "react-router-dom";

import Sidebar from "../components/Sidebar";
import Card from "../components/Card";

import { getDashboard } from "../services/api";

function Dashboard() {

  const [dashboard, setDashboard] = useState({
    totalRecordings: 0,
    latest: []
  });

  const [loading, setLoading] = useState(true);

  const [user, setUser] = useState({
    name: "",
    email: ""
  });

  useEffect(() => {

    const loadDashboard = async () => {

      try {

        const data = await getDashboard();

        if (data.success) {

          setDashboard({
            totalRecordings: data.totalRecordings,
            latest: data.latest
          });

        }

        const storedUser = JSON.parse(
          localStorage.getItem("user")
        );

        if (storedUser) {

          setUser(storedUser);

        }

      }

      catch (err) {

        console.log(err);

      }

      finally {

        setLoading(false);

      }

    };

    loadDashboard();

  }, []);

  return (

    <div className="flex min-h-screen bg-slate-50">

      <Sidebar />

      <main className="flex-1 p-6 md:p-10">

        <motion.div

          initial={{ opacity: 0, y: -20 }}

          animate={{ opacity: 1, y: 0 }}

        >

          <div className="bg-gradient-to-r from-blue-600 to-teal-500 rounded-3xl p-8 text-white">

            <h1 className="text-3xl font-bold">

              Welcome back, {user.name || "User"} 👋

            </h1>

            <p className="mt-2">

              VoiceCare PD Dashboard

            </p>

          </div>

        </motion.div>

        {/* Stats */}

        <section className="grid md:grid-cols-3 gap-6 mt-8">

          <Card>

            <div className="flex items-center gap-4">

              <div className="bg-blue-100 p-4 rounded-full">

                <User className="text-blue-600" />

              </div>

              <div>

                <h3 className="font-bold">

                  {user.name || "User"}

                </h3>

                <p className="text-gray-500">

                  {user.email}

                </p>

              </div>

            </div>

          </Card>

          <Card>

            <div className="flex items-center gap-4">

              <div className="bg-green-100 p-4 rounded-full">

                <FileAudio className="text-green-600" />

              </div>

              <div>

                <h2 className="text-3xl font-bold">

                  {

                    loading

                      ? "..."

                      : dashboard.totalRecordings

                  }

                </h2>

                <p className="text-gray-500">

                  Total Recordings

                </p>

              </div>

            </div>

          </Card>

          <Card>

            <div className="flex items-center gap-4">

              <div className="bg-purple-100 p-4 rounded-full">

                <History className="text-purple-600" />

              </div>

              <div>

                <h2 className="text-3xl font-bold">

                  {

                    loading

                      ? "..."

                      : dashboard.latest.length

                  }

                </h2>

                <p className="text-gray-500">

                  Recent Recordings

                </p>

              </div>

            </div>

          </Card>

        </section>

        {/* Quick Actions */}

        <section className="mt-10">

          <h2 className="text-2xl font-bold">

            Quick Actions

          </h2>

          <div className="grid md:grid-cols-3 gap-6 mt-5">

            <Link to="/record">

              <Card className="cursor-pointer hover:scale-105 transition">

                <Mic

                  size={40}

                  className="text-blue-600"

                />

                <h3 className="font-semibold mt-4">

                  Record Voice

                </h3>

              </Card>

            </Link>

            <Link to="/upload">

              <Card className="cursor-pointer hover:scale-105 transition">

                <Upload

                  size={40}

                  className="text-green-600"

                />

                <h3 className="font-semibold mt-4">

                  Upload Audio

                </h3>

              </Card>

            </Link>

            <Link to="/history">

              <Card className="cursor-pointer hover:scale-105 transition">

                <History

                  size={40}

                  className="text-purple-600"

                />

                <h3 className="font-semibold mt-4">

                  View History

                </h3>

              </Card>

            </Link>

          </div>

        </section>

        {/* Recent Recordings */}

        <section className="mt-10">

          <h2 className="text-2xl font-bold mb-5">

            Recent Recordings

          </h2>

          <Card>

            {

              dashboard.latest.length === 0

              ?

              <p className="text-gray-500 text-center">

                No recordings available

              </p>

              :

              dashboard.latest.map((item) => (

                <div

                  key={item._id}

                  className="border-b py-3 last:border-none flex justify-between"

                >

                  <div>

                    <h4 className="font-semibold">

                      {item.fileName}

                    </h4>

                    <p className="text-gray-500 text-sm">

                      {new Date(item.createdAt).toLocaleDateString()}

                    </p>

                  </div>

                  <div>

                    {item.duration}

                  </div>

                </div>

              ))

            }

          </Card>

        </section>

      </main>

    </div>

  );

}

export default Dashboard;