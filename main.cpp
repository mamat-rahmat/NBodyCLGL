#include <windows.h>
#include <iostream>
#include <QGuiApplication>
#include <QOpenGLWindow>
#include <QOpenGLFunctions_3_3_Compatibility>
#include <QTimer>

#include <boost/compute.hpp>
#include <boost/compute/interop/opengl.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>


const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void updateVelocity(__global const float4* position, __global float4* velocity, float dt, uint N)
    {
        uint gid = get_global_id(0);

        float4 r = { 0.0f, 0.0f, 0.0f, 0.0f };
        float f = 0.0f;
        for(uint i = 0; i != gid; i++) {
            if(i != gid) {
                r = position[i]-position[gid];
                f = length(r)+0.001f;
                f *= f*f;
                f = dt/f;
                velocity[gid] += f*r;
            }
        }
    }
    __kernel void updatePosition(__global float4* position, __global const float4* velocity, float dt)
    {
        uint gid = get_global_id(0);

        position[gid].xyz += dt*velocity[gid].xyz;
    }
);



class NBodyWindow : public QOpenGLWindow,
                    protected QOpenGLFunctions_3_3_Compatibility
{
    Q_OBJECT

public:
    NBodyWindow(std::size_t particles, float dt);
    ~NBodyWindow();

    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void updateParticles();

private:
    QTimer* timer;

    boost::compute::context m_context;
    boost::compute::command_queue m_queue;
    boost::compute::program m_program;
    boost::compute::opengl_buffer m_position;
    boost::compute::vector<boost::compute::float4_>* m_velocity;
    boost::compute::kernel m_velocity_kernel;
    boost::compute::kernel m_position_kernel;

    bool m_initial_draw;

    const boost::compute::uint_ m_particles;
    const float m_dt;
};

NBodyWindow::NBodyWindow(std::size_t particles, float dt)
    : m_initial_draw(true), m_particles(particles), m_dt(dt)
{
    // create a timer to redraw as fast as possible
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(update()));
    timer->start(1);
}

NBodyWindow::~NBodyWindow()
{
    delete m_velocity;

    // delete the opengl buffer
    GLuint vbo = m_position.get_opengl_object();
    glDeleteBuffers(1, &vbo);
}

void NBodyWindow::initializeGL()
{
    initializeOpenGLFunctions();

    // create context, command queue and program
    m_context = boost::compute::opengl_create_shared_context();
    m_queue = boost::compute::command_queue(m_context, m_context.get_device());
    m_program = boost::compute::program::create_with_source(source, m_context);
    m_program.build();

    // prepare random particle positions that will be transferred to the vbo
    boost::compute::float4_* temp = new boost::compute::float4_[m_particles];
    boost::random::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    boost::random::mt19937_64 gen;
    for(size_t i = 0; i < m_particles; i++) {
        temp[i][0] = dist(gen);
        temp[i][1] = dist(gen);
        temp[i][2] = dist(gen);
        temp[i][3] = 1.0f;
    }

    // create an OpenGL vbo
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m_particles*sizeof(boost::compute::float4_), temp, GL_DYNAMIC_DRAW);

    // create a OpenCL buffer from the vbo
    m_position = boost::compute::opengl_buffer(m_context, vbo);
    delete[] temp;

    // create buffer for velocities
    m_velocity = new boost::compute::vector<boost::compute::float4_>(m_particles, m_context);
    boost::compute::fill(m_velocity->begin(), m_velocity->end(), boost::compute::float4_(0.0f, 0.0f, 0.0f, 0.0f), m_queue);

    // create compute kernels
    m_velocity_kernel = m_program.create_kernel("updateVelocity");
    m_velocity_kernel.set_arg(0, m_position);
    m_velocity_kernel.set_arg(1, m_velocity->get_buffer());
    m_velocity_kernel.set_arg(2, m_dt);
    m_velocity_kernel.set_arg(3, m_particles);
    m_position_kernel = m_program.create_kernel("updatePosition");
    m_position_kernel.set_arg(0, m_position);
    m_position_kernel.set_arg(1, m_velocity->get_buffer());
    m_position_kernel.set_arg(2, m_dt);
}
void NBodyWindow::resizeGL(int width, int height)
{
    // update viewport
    glViewport(0, 0, width, height);
}
void NBodyWindow::paintGL()
{
    // clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // check if this is the first draw
    if(m_initial_draw) {
        // do not update particles
        m_initial_draw = false;
    } else {
        // update particles
        updateParticles();
    }

    // draw
    glVertexPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, m_particles);
    glFinish();
}
void NBodyWindow::updateParticles()
{
    // enqueue kernels to update particles and make sure that the command queue is finished
    boost::compute::opengl_enqueue_acquire_buffer(m_position, m_queue);
    m_queue.enqueue_1d_range_kernel(m_velocity_kernel, 0, m_particles, 0).wait();
    m_queue.enqueue_1d_range_kernel(m_position_kernel, 0, m_particles, 0).wait();
    m_queue.finish();
    boost::compute::opengl_enqueue_release_buffer(m_position, m_queue);
}

int main(int argc, char *argv[])
{
    boost::compute::uint_ particles = 10000;
    float dt = 0.0001f;
    QGuiApplication app(argc, argv);
    NBodyWindow nbody(particles, dt);

    nbody.show();

    return app.exec();
}

#include "main.moc"
